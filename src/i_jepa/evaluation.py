import os
import sys; sys.path.append('.')
import torch
import torch.nn as nn
from src.i_jepa.models import EncoderViT
from pathlib import Path

checkpoint_dir = Path(".")

saved_dict = torch.load('checkpoint_ep0.pt')
print(saved_dict['target_encoder'])