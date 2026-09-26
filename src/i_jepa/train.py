import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
import os
import sys; sys.path.append(".")
from config import Config
import logging
from src.i_jepa.data_prepare import install_data_folder_tiny, Make_transform, Mask_collator
from src.i_jepa.models import EncoderViT, PredictorViT

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

# --------------------------------------------------------------------------------
def main():
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
        pin_memory=cfg.pin_mem, # in future code xb = xb.to('cuda', non_blocking=True)
        persistent_workers=False,
    )
# --------------------------------------------------------------------------------
    device = 'cpu'
    if torch.cuda.is_available():
        device = cfg.device
    print(f"using device: {device}")
    encoder_vit_context = EncoderViT(cfg)
    predictor_vit = PredictorViT(cfg)
    encoder_vit_context.to(device)
    predictor_vit.to(device)
    
    for epoch in range(cfg.num_epochs):
        dist_sampler.set_epoch(epoch)
        for step, (xb, context_indecies, targets_indecies) in enumerate(data_loader):
            s_x = encoder_vit_context(xb, context_indecies)
            print(f"{s_x.shape} from s_x data_loader")

            s_y_pred = predictor_vit(s_x, context_indecies, targets_indecies)
            print(f"{s_y_pred.shape} from s_y_pred data_loader")
            print(f"{step=}")
            # Target branch
            # with.torch.no_grad:
                #   s_y = encoder_vit_context(xb)
                #   s_y = apply_masks(s_y)
            
            break

if __name__ == "__main__":
    main()