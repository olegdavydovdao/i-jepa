import torch
import torch.nn as nn
import sys; sys.path.append(".")
import numpy as np


class PatchEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Conv2d(cfg.num_img_channels, cfg.emb_dims, kernel_size=cfg.patch_size, stride=cfg.patch_size)
    def forward(self, x): # x is (B,C,H,W) or (2,3,224,224) | 3, 768, 16,16 above
        x = self.proj(x).flatten(2).transpose(1,2)
        return x # (B, num_pathces, C)

def get_2d_sincos_pos_embed(cfg):
    # compute on CPU by default
    h_range = torch.arange(cfg.height, dtype=torch.float32)
    w_range = torch.arange(cfg.width, dtype=torch.float32)
    grid = torch.meshgrid(h_range, w_range, indexing='ij')
    grid = torch.stack(grid)
    pos_embed = get_2d_sincos_pos_embed_from_grid(cfg.emb_dims, grid)
    return pos_embed # (N, D)

def get_2d_sincos_pos_embed_from_grid(emb_dims, grid):
    assert emb_dims % 2 == 0
    half_emb_dims = emb_dims//2
    emb_h = get_1d_sincos_pos_embed_from_grid(half_emb_dims, grid[0]) # (N, D//2)
    emb_w = get_1d_sincos_pos_embed_from_grid(half_emb_dims, grid[1]) # (N, D//2)
    emb = torch.cat((emb_h, emb_w), dim=1) # (N, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(half_emb_dims, pos):
    pos = pos.view(-1,1) # (N, 1)
    assert half_emb_dims % 2 == 0
    quarter_emb_dims = half_emb_dims//2
    omega = torch.arange(quarter_emb_dims, dtype=torch.float32).view(1,-1)
    omega /= quarter_emb_dims
    omega = 1.0 / 10000**omega # (1, D//4)
    out = pos * omega # (N, D//4)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat((emb_sin, emb_cos), dim=1) # (N, D//2)
    return emb

class Attention(nn.Module):
    pass

class MLP(nn.Module):
    pass

class Block(nn.Module):
    pass

class EncoderViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.emb_dims), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(cfg) # (N,D) | on CPU
        with torch.no_grad():
            self.pos_embed.copy_(pos_embed.unsqueeze(0)) # copy on GPU
    def forward(self, x, masks=None): # x is (B,3,224,224)
        x = self.patch_embed(x) # (B, T, C)
        return x

class PredictorViT(nn.Module):
    pass
