import torch
import torch.nn as nn
from torch.nn import functional as F
import sys; sys.path.append(".")
import numpy as np
from functools import partial


class PatchEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Conv2d(cfg.num_img_channels, cfg.emb_dims, kernel_size=cfg.patch_size, stride=cfg.patch_size)
    def forward(self, x): # x is (B,C,H,W) or (2,3,224,224) | 3, 768, 16,16 above
        x = self.proj(x).flatten(2).transpose(1,2)
        return x # (B, num_pathces, C)

def get_2d_sincos_pos_embed(cfg, emb_dims):
    # compute on CPU by default
    h_range = torch.arange(cfg.height, dtype=torch.float32)
    w_range = torch.arange(cfg.width, dtype=torch.float32)
    grid = torch.meshgrid(h_range, w_range, indexing='ij')
    grid = torch.stack(grid)
    pos_embed = get_2d_sincos_pos_embed_from_grid(emb_dims, grid)
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
    def __init__(self, cfg, emb_dim_any):
        super().__init__()
        assert emb_dim_any % cfg.num_heads == 0
        self.qkv = nn.Linear(emb_dim_any, 3*emb_dim_any, bias=cfg.qkv_bias)
        self.proj = nn.Linear(emb_dim_any, emb_dim_any)
        self.proj.FLAG_SCALE_INIT_RESIDUAL = 1
        self.num_heads = cfg.num_heads
        self.head_dim = emb_dim_any // self.num_heads
        self.emb_dims = emb_dim_any

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x) # (B,T,3C)
        q,k,v = qkv.split(self.emb_dims, dim=2) # each (B,T,C)
        q,k,v = [i.view(B,T, self.num_heads, self.head_dim).transpose(1,2) for i in [q,k,v]]
        # q,k,v are each (B,n_h,T,H)
        # scaled_dot_product_attention: att = q@k*scale -> att = softmax(dim=-1) -> att@v
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False) # (B,n_h,T,H)
        x = x.transpose(1,2).contiguous().view(B,T,C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, cfg, emb_dim_any):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim_any, cfg.mlp_expander*emb_dim_any)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(cfg.mlp_expander*emb_dim_any, emb_dim_any)
        self.proj.FLAG_SCALE_INIT_RESIDUAL = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, cfg, emb_dim_any, layer_norm):
        super().__init__()
        self.ln_1 = layer_norm(emb_dim_any)
        self.attn = Attention(cfg, emb_dim_any)
        self.ln_2 = layer_norm(emb_dim_any)
        self.mlp = MLP(cfg, emb_dim_any)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def apply_masks(x, masks): # x = (B,N,D)
    all_x = []
    for m_tok_keep in masks: # m_tok_keep == (B, restrict_num_patches)
        m_tok_keep = m_tok_keep.unsqueeze(-1).repeat(1,1,x.shape[-1]) # (B, restrict_num_patches,D)
        all_x += [torch.gather(x, dim=1, index=m_tok_keep)]
    return torch.cat(all_x, dim=0)

def init_weights_shared(module, std, depth_any):

    def wei_bias_init(std):
        nn.init.normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    if isinstance(module, nn.Linear):
        if hasattr(module, 'FLAG_SCALE_INIT_RESIDUAL'):
            std*=(2*depth_any)**-0.5
        wei_bias_init(std=std)
    elif isinstance(module, nn.Conv2d):
        wei_bias_init(std=std)

class EncoderViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_patches = cfg.num_patches
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.emb_dims), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(cfg, cfg.emb_dims) # (N,D) | on CPU
        with torch.no_grad():
            self.pos_embed.copy_(pos_embed.unsqueeze(0)) # copy on GPU
        layer_norm = partial(nn.LayerNorm, eps=cfg.eps_layer_norm)
        self.blocks = nn.ModuleList([Block(cfg, emb_dim_any=cfg.emb_dims, layer_norm=layer_norm) for _ in range(cfg.depth)])
        self.ln_f = layer_norm(cfg.emb_dims)
        self.init_std = cfg.init_std
        self.depth = cfg.depth
        self.apply(self._init_weights)

    def _init_weights(self, module):
        init_weights_shared(module, std=self.init_std, depth_any=self.depth)

    def forward(self, x, context_indecies=None): # x is (B,3,224,224)
        x = self.patch_embed(x) # (B, N, D)
        B, N, D = x.shape
        assert N == self.num_patches, f"(N={N}) != (num_patches={self.num_patches})"
        x = x + self.pos_embed # (B, N, D) = (B, N, D) + (1, N, D)

        # restrict x only to allowable tokens
        if context_indecies is not None:
            x = apply_masks(x, masks=context_indecies) # (B, N_restrict, D) restict in this case context

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # final layer norm
        x = self.ln_f(x)
        return x

class PredictorViT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.predictor_embed = nn.Linear(cfg.emb_dims, cfg.pred_emb_dims)
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.pred_emb_dims), requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(cfg, cfg.pred_emb_dims)
        with torch.no_grad():
            self.predictor_pos_embed.copy_(predictor_pos_embed.unsqueeze(0))
        self.init_std = cfg.init_std
        self.mask_token = nn.Parameter(torch.randn(1,1,cfg.pred_emb_dims)*self.init_std)
        layer_norm = partial(nn.LayerNorm, eps=cfg.eps_layer_norm)
        self.pred_depth = cfg.pred_depth
        self.predictor_blocks = nn.ModuleList([Block(cfg, emb_dim_any=cfg.pred_emb_dims, layer_norm=layer_norm) for _ in range(self.pred_depth)])
        self.predictor_ln_f = layer_norm(cfg.pred_emb_dims)
        self.predictor_proj = nn.Linear(cfg.pred_emb_dims, cfg.emb_dims)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        init_weights_shared(module, std=self.init_std, depth_any=self.pred_depth)

    def forward(self, x, context_indecies, targets_indecies):
        assert (context_indecies is not None) and (targets_indecies is not None), 'context and target indecies are needed'
        B = x.shape[0]
        x = self.predictor_embed(x)
        pos_embed_cont = self.predictor_pos_embed.repeat(B,1,1) # (B,N,D)
        pos_embed_cont = apply_masks(pos_embed_cont, masks=context_indecies) # (B,N_lim_cont,D)
        x = x + pos_embed_cont
        N_lim_cont, D = x.shape[1], x.shape[2]
        
        pos_embed_target = self.predictor_pos_embed.repeat(B,1,1) # (B,N,D)
        pos_embed_target = apply_masks(pos_embed_target, masks=targets_indecies) # (4*B, N_lim_target, D)
        pred_tokens = pos_embed_target + self.mask_token

        x = x.repeat(len(targets_indecies),1,1) # B -> 4B
        x = torch.cat([x, pred_tokens], dim=1) # (4B, N_lim_cont + N_lim_target, D)

        for block in self.predictor_blocks:
            x = block(x)
        
        x = x[:, N_lim_cont:]
        x = self.predictor_ln_f(x)
        x = self.predictor_proj(x)
        return x
        