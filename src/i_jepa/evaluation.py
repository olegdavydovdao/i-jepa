import os
import sys; sys.path.append('.')
import torch
import torch.nn as nn
from src.i_jepa.models import EncoderViT
from pathlib import Path

path = os.path.join(".", "checkpoints")
checkpoint_dir = Path(path)
checkpoint_files = list(checkpoint_dir.glob("checkpoint_ep*.pt"))
final_file = checkpoint_files[-1]
saved_dict = torch.load(final_file)
target_encoder_params = saved_dict['target_encoder']