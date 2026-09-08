import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_from_disk, load_dataset, Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os
import sys; sys.path.append(".")
from config import Config
torch.manual_seed(0)

# Install small subset of Imagenet1k
# Note: for full Imagenet code slightly will change due without .save_to_disk()
def install_data_folder_tiny(split):
    assert split in ["train", "validation"], 'argument in install_data_folder() should be: "train" or "validation"'
    print(f'Data folder {split} is missing, starting download "tiny" Imagenet1k {split} set from Hugging Face')
    stream = load_dataset('ILSVRC/imagenet-1k', split=split, streaming=True) # IterableDataset
    if split == "train":
        subset_stream = stream.take(cfg.num_train_rows)
    else:
        subset_stream = stream.take(cfg.num_val_rows)
    local_dataset = Dataset.from_generator(lambda: iter(subset_stream))
    split_path = f"{cfg.tiny_data_folder_name}/{split}"
    local_dataset.save_to_disk(split_path)

# n_rows with raw images -> n_rows with 224x224 crop tensor for each image independently
class Make_transform():
    def __init__(self):
        self.transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(cfg.crop_size, scale=cfg.crop_scale),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cfg.normalization[0], std=cfg.normalization[1])
        ])
    def __call__(self, n_rows):
        n_rows["image"] = [self.transform(img) for img in n_rows["image"]]
        return n_rows

# Converting the HF data and mask strategy
class Mask_collator():
    def __init__(self):
        self.num_patches = (cfg.crop_size//cfg.patch_size)**2 # int 196
    def __call__(self, list_of_i1l1_dicts):
        # get xb
        list_images = [il_dict["image"] for il_dict in list_of_i1l1_dicts]
        xb = torch.stack(list_images)

        # src/masks/multiblock.py
        # mask strategy for single gpu. for ddp change the code.
        # for target blocks T1 != T_m or T1 == T_m?
        # torch.randint(self.num_patches)
        context_patch_indices = None
        target_patch_indices = None
        return xb#, context_patch_indices, target_patch_indices

# --------------------------------------------------------------------------------

cfg = Config()
# ImageNet_tiny data installation at 1st run
if os.path.isdir(cfg.tiny_data_folder_name):
    pass
else:
    install_data_folder_tiny("train")
    install_data_folder_tiny("validation")

# Load and pre-transform train data
train_data = load_from_disk(f"{cfg.tiny_data_folder_name}/train")
make_transform = Make_transform()
def wrap_transform(n_rows): # to num_workers works fine
    return make_transform(n_rows)
train_data = train_data.with_transform(wrap_transform)
mask_collator = Mask_collator()

data_loader = DataLoader(
    train_data,
    batch_size=cfg.batch_size,
    collate_fn=mask_collator
)
# shuffle or sampler(<- i need it): how it happens and how do it to (image,label) not destroy?
# num_workers
# collate_fn in DataLoader is the thing that mentioned in masking strategy ijepa paper page 12 (gemini said).

for xb in data_loader:
    print(xb.shape)
    # print(xb)
    break