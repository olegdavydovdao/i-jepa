from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    crop_size: int = 224
    crop_scale: tuple = (0.3, 1.0)
    normalization: tuple = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    tiny_data_folder_name: str = "imagenet1k_tiny"
    num_train_rows: int = 1000
    num_val_rows: int = 100