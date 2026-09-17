import torch
import torch.nn as nn
import sys; sys.path.append(".")


class PatchEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Conv2d(cfg.num_img_channels, cfg.emb_dims, kernel_size=cfg.patch_size, stride=cfg.patch_size)
    def forward(self, x): # x is (B,C,H,W) or (2,3,224,224) | 3, 768, 16,16 above
        x = self.proj(x).flatten(2).transpose(1,2)
        return x # (B, num_pathces, C)

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
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbed(cfg)
    def forward(self, x, masks=None):
        # if masks is not None:
        #     if not isinstance(masks, list):
        #         masks = [masks]
        x = self.patch_embed(x)
        return x

class PredictorViT(nn.Module):
    pass
