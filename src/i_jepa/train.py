import os
import sys; sys.path.append(".")
import torch
from torch.nn import functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from config import Config
import logging
from src.i_jepa.data_prepare import install_data_folder_tiny, Make_transform, Mask_collator
from src.i_jepa.models import EncoderViT, PredictorViT, apply_masks
from src.i_jepa.shedulers import LRScheduler, WDScheduler
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# --------------------------------------------------------------------------------
def main():
    use_ddp = int(os.environ.get('RANK', -1)) != -1
    if use_ddp:
        assert torch.cuda.is_available(), 'need cuda to DDP'
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        init_process_group(backend='nccl')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = f'cuda:{local_rank}'
        else: device = 'cpu'
    master_process = rank == 0
    if master_process:
        print(f"{use_ddp=}")

    # to set num_workers creation in DataLoader, not effect on torchrun processes
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    cfg = Config()
    # ImageNet_tiny data installation at 1st run
    if os.path.isdir(cfg.tiny_data_folder_name):
        pass
    else:
        install_data_folder_tiny(cfg, "train")
        install_data_folder_tiny(cfg, "validation")

    # Load and pre-transform train data
    train_data = load_from_disk(f"{cfg.tiny_data_folder_name}/train")
    make_transform = Make_transform(cfg)
    train_data = train_data.with_transform(make_transform)
    mask_collator = Mask_collator(cfg)

    # for DDP training | it has own shuffle=True
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_data,
        num_replicas=cfg.world_size,
        rank=cfg.rank,
        shuffle = True,
    )

    data_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        collate_fn=mask_collator,
        num_workers=cfg.num_workers,
        sampler = dist_sampler,
        drop_last=cfg.drop_last_data,
        pin_memory=cfg.pin_mem, # relate with non_blocking=True
        persistent_workers=False,
    )
# --------------------------------------------------------------------------------
    encoder_vit_context = EncoderViT(cfg)
    predictor_vit = PredictorViT(cfg)
    encoder_vit_context.to(device)
    predictor_vit.to(device)
    target_encoder_vit = copy.deepcopy(encoder_vit_context) # already on cuda
    if use_ddp:
        encoder_vit_context = DDP(encoder_vit_context, device_ids=[local_rank])
        predictor_vit = DDP(predictor_vit, device_ids=[local_rank])
        target_encoder_vit = DDP(target_encoder_vit, device_ids=[local_rank])
    for p in target_encoder_vit.parameters():
        p.requires_grad = False

    def params_generator(model):
        params_2d = (p for n,p in model.named_parameters() if p.requires_grad and p.dim() >= 2)
        params_1d = (p for n,p in model.named_parameters() if p.requires_grad and p.dim() < 2)
        return params_2d, params_1d
    
    encoder_params_2d, encoder_params_1d = params_generator(encoder_vit_context)
    predictor_params_2d, predictor_params_1d = params_generator(predictor_vit)
    param_groups = [
        # 2d
        {'params': encoder_params_2d},
        {'params': predictor_params_2d},
        # 1d
        {'params': encoder_params_1d, 'weight_decay': 0.0},
        {'params': predictor_params_1d, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(param_groups, fused=cfg.use_adamw_fused) # lr betas

    i_per_ep = len(data_loader)
    ema = cfg.ema
    momentum_scheduler = (ema[0] + (ema[1]-ema[0])*i/(i_per_ep*cfg.num_epochs)
                          for i in range(i_per_ep*cfg.num_epochs))

    get_lr = LRScheduler(cfg, i_per_ep)
    get_wd = WDScheduler(cfg, i_per_ep)

    # torch.compile before DDP
    # DDP
    # set_float32_matmul_precision
    # norm
    # time

    for epoch in range(cfg.num_epochs):
        dist_sampler.set_epoch(epoch)

        for step, (xb, context_indecies, targets_indecies) in enumerate(data_loader):
            optimizer.zero_grad()

            # data to device
            xb = xb.to(device, non_blocking=True)
            context_indecies = [m.to(device, non_blocking=True) for m in context_indecies]
            targets_indecies = [m.to(device, non_blocking=True) for m in targets_indecies]

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=cfg.use_bfloat16):
                # context branch forward
                s_x = encoder_vit_context(xb, context_indecies)
                s_y_pred = predictor_vit(s_x, context_indecies, targets_indecies)

                # target branch forward
                with torch.no_grad():
                    s_y = target_encoder_vit(xb)
                    s_y = F.layer_norm(s_y, (s_y.shape[-1],), eps=cfg.eps_layer_norm)
                    s_y = apply_masks(s_y, targets_indecies)

                # get loss
                loss = F.smooth_l1_loss(s_y_pred, s_y)
            print(f"{step} | {loss=}")
            if step == 50:
                break

            # update context branch models
            loss.backward()
            lr = get_lr.step()
            wd = get_wd.step()
            for group in optimizer.param_groups:
                group['lr'] = lr
                if group['weight_decay'] != 0.0: # default = 0.01
                    group['weight_decay'] = wd
            optimizer.step()

            # EMA update target branch
            with torch.no_grad():
                m = next(momentum_scheduler)
                for p_c, p_t in zip(encoder_vit_context.parameters(), target_encoder_vit.parameters()):
                    if p_c.requires_grad:
                        p_t.mul_(m).add_(p_c, alpha=1.0-m)
            

if __name__ == "__main__":
    main()