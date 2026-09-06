import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_from_disk, load_dataset, Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os
import sys

# Hyperparameters
torch.manual_seed(0)
crop_size = 224,
crop_scale = (0.3, 1.0)
normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
data_folder_name = "imagenet1k_tiny"
num_train_rows = 1000
num_val_rows = 100

# Install small subset of Imagenet1k
# Note: for full Imagenet code slightly will change due without .save_to_disk())
def install_data_folder_tiny(split):
    assert split in ["train", "validation"], 'argument in install_data_folder() should be: "train" or "validation"'
    print(f'Data folder {split} is missing, starting download "tiny" Imagenet1k {split} set from Hugging Face')
    stream = load_dataset('ILSVRC/imagenet-1k', split=split, streaming=True) # IterableDataset
    if split == "train":
        subset_stream = stream.take(num_train_rows)
    else:
        subset_stream = stream.take(num_val_rows)
    local_dataset = Dataset.from_generator(lambda: iter(subset_stream))
    split_path = f"{data_folder_name}/{split}"
    local_dataset.save_to_disk(split_path)

# n_rows raw images -> n_rows with 224x224 crop tensor for each image independently
class Make_transform():
    def __init__(self, crop_size, crop_scale, normalization):
        self.transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(crop_size, scale=crop_scale),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalization[0], std=normalization[1])
        ])
    def __call__(self, n_rows):
        n_rows["image"] = [self.transform(img) for img in n_rows["image"]]
        return n_rows

# Converting the HF data and mask strategy
class Mask_collator():
    def __init__(self): # HF Dataset.__getitem__ change the HF type of data
        pass
    def __call__(self, list_of_i1l1_dicts):
        list_images = [il_dict["image"] for il_dict in list_of_i1l1_dicts]
        xb = torch.stack(list_images)
        return xb
# ----

# ImageNet_tiny data installation at 1st run
if os.path.isdir(data_folder_name):
    pass
else:
    install_data_folder_tiny("train")
    install_data_folder_tiny("validation")

# Load and pre-transform train data
train_data = load_from_disk(f"{data_folder_name}/train")
make_transform = Make_transform(crop_size, crop_scale, normalization)
def wrap_transform(n_rows): # to num_workers works fine
    return make_transform(n_rows)
train_data = train_data.with_transform(wrap_transform)
mask_collator = Mask_collator()

data_loader = DataLoader(
    train_data,
    batch_size=2,
    collate_fn=mask_collator
)
# shuffle or sampler(<- i need it): how it happens and how do it to (image,label) not destroy?
# num_workers
# collate_fn in DataLoader is the thing that mentioned in masking strategy ijepa paper page 12 (gemini said).

for xb in data_loader:
    print(xb.shape)
    print(xb)
    break