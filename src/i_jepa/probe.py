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
    
    path_to_saved_dir = "checkpoints"
    path_to_saved_file = "checkpoint_end_pretrain.pt"
    path_to_pt = os.path.join(path_to_saved_dir, path_to_saved_file)
    saved_dict = torch.load(path_to_pt)
    saved_target_encoder = saved_dict['target_encoder']

    target_encoder_vit = EncoderViT(cfg)
    target_encoder_vit.load_state_dict(saved_target_encoder)
    for p in target_encoder_vit.parameters():
        p.requires_grad = False
    target_encoder_vit.to(device)

    major, minor = torch.cuda.get_device_capability()
    use_compile = cfg.use_compile and major >= 7
    # compile forward function of 3 separate models.
    if master_process:
        print(f"{use_compile=}")
    if use_compile:
        target_encoder_vit = torch.compile(target_encoder_vit)

if __name__ == '__main__':
    main()

# save later
# 'probe': probe.state_dict(),
# 'probe_optimizer': probe_optimizer.state_dict(),