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

def main():
    pass

if __name__ == '__main__':
    main()