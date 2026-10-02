from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Data
    crop_size: tuple = (224, 224)
    crop_scale: tuple = (0.3, 1.0)
    normalization: tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tiny_data_folder_name: str = "imagenet1k_tiny"
    num_train_rows: int = 1000
    num_val_rows: int = 100
    batch_size: int = 2
    num_workers: int = 2
    pin_mem: bool = True
    drop_last_data: bool = True
    
    # Mask
    patch_size: int = 16
    height: int = crop_size[0]//patch_size # 14
    width: int = crop_size[1]//patch_size # 14
    num_patches: int = height*width # 196
    allow_overlap: bool = False
    min_mask_num_patches = 10
    ## targets masks
    target_aspect_ratio_range: tuple = (0.75, 1.5) # h/w
    target_mask_scale_range: tuple = (0.15, 0.2)
    num_target_masks: int = 4
    ## context mask
    context_aspect_ratio_range: tuple = (1.0, 1.0)
    context_mask_scale_range: tuple = (0.85, 1.0)
    num_context_masks: int = 1

    # Models: ViT-B/16 86M for debug | tiny 5.7M or small 22M for training
    num_heads: int = 12
    mlp_expander: int = 4
    qkv_bias: bool = True
    eps_layer_norm: float = 1e-6
    init_std = 0.02
    ## Enocder
    emb_dims: int = 768
    num_img_channels: int = 3
    depth: int = 12
    ## Predictor
    pred_emb_dims: int = 384
    pred_depth: int = 6

    # Optimization
    ema: tuple = (0.996, 1.0)
    num_epochs: int = 1
    start_lr: float = 2e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-6
    start_wd: float = 0.04 # weight_decay
    final_wd: float = 0.4
    warmup_over_epochs: float = 40/300

    use_adamw_fused: bool = True
    ## cuda | GPU == A100 for future
    world_size: int = 1 # to enable DDP == 2
    rank: int = 0
    use_bfloat16: bool = False