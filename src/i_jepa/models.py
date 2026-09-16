import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys; sys.path.append(".")
from config import Config
from src.i_jepa import data_prepare

class PatchEmbed(nn.Module):
    pass

def get_2d_sincos_pos_embed():
    pass

def get_2d_sincos_pos_embed_from_grid():
    pass

class Attention(nn.Module):
    pass

class MLP(nn.Module):
    pass

class Block(nn.Module):
    pass

class EncoderViT(nn.Module):
    pass

class PredictorViT(nn.Module):
    pass
