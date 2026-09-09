from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Data
    crop_size: int = 224
    crop_scale: tuple = (0.3, 1.0)
    normalization: tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tiny_data_folder_name: str = "imagenet1k_tiny"
    num_train_rows: int = 1000
    num_val_rows: int = 100
    batch_size: int = 2
    num_workers: int = 2
    # pin_mem: bool = True
    
    # Mask
    patch_size: int = 16
    allow_overlap: bool = False
    # targets masks
    target_aspect_ratio: tuple = (0.75, 1.5)
    target_mask_scale: tuple = (0.15, 0.2)
    num_target_masks: int = 4
    # context mask
    context_aspect_ratio: tuple = (1.0, 1.0)
    context_mask_scale: tuple = (0.85, 1.0)
    num_context_masks: int = 1
    # min_num_patches_context_block: int = 10

    # Model

    # Optimization