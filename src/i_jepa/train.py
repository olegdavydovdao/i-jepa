import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
import os
import sys; sys.path.append(".")
from config import Config
from logging import getLogger
from src.i_jepa.data_prepare import install_data_folder_tiny, Make_transform, Mask_collator

# --------------------------------------------------------------------------------
logger = getLogger()
torch.manual_seed(0)
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

# --------------------------------------------------------------------------------

if __name__=="__main__":
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
        drop_last=True,
        pin_memory=True, # in future code xb = xb.to('cuda', non_blocking=True)
        persistent_workers=False,
    )

    for epoch in range(cfg.num_epochs):
        dist_sampler.set_epoch(epoch) 
        for xb, context_indecies, targets_indecies in data_loader:
            # print(xb.shape)
            # print(context_indecies)
            # print()
            # for k in range(len(targets_indecies)):
            #     print(targets_indecies[k])
            print(xb)
            break