import os
import sys; sys.path.append(".")
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from config import Config
from src.i_jepa.data_prepare import Make_transform, Mask_collator
from src.i_jepa.models import EncoderViT
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from src.i_jepa.train import main as train_main

def main():
    share_to_train = train_main(share=True)
    use_ddp, rank, local_rank, world_size, device, master_process, device_type, make_transform, mask_collator, dist_sampler, data_loader, cfg, batch_size_per_process, grad_accum_steps = share_to_train()

if __name__ == '__main__':
    main()