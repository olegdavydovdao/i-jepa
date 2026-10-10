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
import time

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger()

# --------------------------------------------------------------------------------
def main():
    # to set num_workers creation in DataLoader
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # run code via torchrun and non-torchrun.
    use_ddp = int(os.environ.get('RANK', -1)) != -1
    if use_ddp:
        assert torch.cuda.is_available(), 'need cuda to DDP'
        init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = f'cuda:{local_rank}'
        else: device = 'cpu'
    master_process = rank==0
    if master_process:
        print(f"{use_ddp=}")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    cfg = Config()

    # grad accum set up
    assert cfg.total_batch_size % (cfg.batch_size*world_size) == 0, 'output of "%" is not 0'
    batch_size_per_process = cfg.total_batch_size // world_size
    grad_accum_steps = batch_size_per_process // cfg.batch_size
    if master_process:
        print(f"{cfg.total_batch_size=}")
        print(f"{batch_size_per_process=}")
        print(f"{grad_accum_steps=}")

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
        num_replicas=world_size,
        rank=rank,
        shuffle = True,
    )

    data_loader = DataLoader(
        train_data,
        batch_size=batch_size_per_process,
        collate_fn=mask_collator,
        num_workers=cfg.num_workers,
        sampler = dist_sampler,
        drop_last=cfg.drop_last_data,
        pin_memory=cfg.pin_mem, # relate with non_blocking=True
        persistent_workers=False,
    )
# --------------------------------------------------------------------------------
    torch.set_float32_matmul_precision('high') # TF32 if available
    encoder_vit_context = EncoderViT(cfg)
    predictor_vit = PredictorViT(cfg)
    encoder_vit_context.to(device)
    predictor_vit.to(device)
    target_encoder_vit = copy.deepcopy(encoder_vit_context) # already on cuda
    for p in target_encoder_vit.parameters():
        p.requires_grad = False
    if use_ddp:
        with torch.no_grad():
            for param in target_encoder_vit.parameters():
                dist.broadcast(param, src=0)

    major, minor = torch.cuda.get_device_capability()
    use_compile = cfg.use_compile and major >= 7
    # compile forward function of 3 separate models.
    if master_process:
        print(f"{use_compile=}")
    if use_compile:
        encoder_vit_context = torch.compile(encoder_vit_context)
        predictor_vit = torch.compile(predictor_vit)
        target_encoder_vit = torch.compile(target_encoder_vit)

    if use_ddp:
        # within DDP init dist.broadcast are used.
        encoder_vit_context = DDP(encoder_vit_context, device_ids=[local_rank])
        predictor_vit = DDP(predictor_vit, device_ids=[local_rank])
    raw_context_encoder = encoder_vit_context.module if use_ddp else encoder_vit_context

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

    for epoch in range(cfg.num_epochs):
        dist_sampler.set_epoch(epoch)

        for step, (xb_proc, context_indecies_proc, targets_indecies_proc) in enumerate(data_loader):
            t0 = time.time()
            optimizer.zero_grad()
            loss_accum = 0.0
            for k in range(grad_accum_steps):
                # get new next batch_size in batch_size_per_process
                xb = xb_proc[k*cfg.batch_size: k*cfg.batch_size + cfg.batch_size]
                context_indecies = [m[k*cfg.batch_size: k*cfg.batch_size + cfg.batch_size] for m in context_indecies_proc]
                targets_indecies = [m[k*cfg.batch_size: k*cfg.batch_size + cfg.batch_size] for m in targets_indecies_proc]

                # data to device
                xb = xb.to(device, non_blocking=True)
                context_indecies = [m.to(device, non_blocking=True) for m in context_indecies]
                targets_indecies = [m.to(device, non_blocking=True) for m in targets_indecies]

                if use_ddp:
                    encoder_vit_context.require_backward_grad_sync = (k==grad_accum_steps-1)
                    predictor_vit.require_backward_grad_sync = (k==grad_accum_steps-1)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=cfg.use_bfloat16):
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
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            if use_ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            # optimize step
            lr = get_lr.step()
            wd = get_wd.step()
            for group in optimizer.param_groups:
                group['lr'] = lr
                if group['weight_decay'] != 0.0: # default = 0.01
                    group['weight_decay'] = wd
            optimizer.step()

            # EMA update target branch
            # do i need raw model from compile and ddp?
            with torch.no_grad():
                m = next(momentum_scheduler)
                for p_c, p_t in zip(raw_context_encoder.parameters(), target_encoder_vit.parameters()):
                    if p_c.requires_grad:
                        p_t.mul_(m).add_(p_c, alpha=1.0-m)

            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            batch_per_sec = cfg.total_batch_size / dt
            if master_process:
                print(f"step: {step:4d} | loss_accum: {loss_accum.item():.4f} | dt: {dt:.2f}s | lr: {lr:.4e} | wd: {wd:.4f} | batch/sec: {batch_per_sec:.2f}")
            if step == 50:
                break

    
    if use_ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()