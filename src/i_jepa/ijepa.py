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

# Install the data only at 1st time
folder_name = "imagenet1k_1000rows"
if os.path.isdir(folder_name):
    print('Data folder is exists')
else:
    install_all_data = False
    print(f'Data folder is missing, starting download from Hugging Face {install_all_data=}')
    if install_all_data:
        # install all data next save them. 2x works with 170GB storage, i don't like it.
        # train_set = load_dataset('ILSVRC/imagenet-1k', split='train', cache_dir=folder_name)
        # train_set.save_to_disk(folder_name)
        pass
    else:
        num_rows = 1000
        train_set_stream = load_dataset('ILSVRC/imagenet-1k', split='train', streaming=True) # IterableDataset
        subset_stream = train_set_stream.take(num_rows)
        local_dataset = Dataset.from_generator(lambda: iter(subset_stream))
        local_dataset.save_to_disk(folder_name)

# Raw imagenet1k image -> 224x224 crop tensor
def make_transform(
    crop_size = 224,
    crop_scale = (0.3, 1.0),
    normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
):
    transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(crop_size, scale=crop_scale),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=normalization[0], std=normalization[1])
    ])
    return transform

def transform_batch(n_rows):
    n_rows["image"] = [transform(img) for img in n_rows["image"]]
    return n_rows

# I need: 1.28M rows in ROM, xb,yb tensors im RAM.
# ImageNet data prepare
train_data = load_from_disk(folder_name)
transform = make_transform()
train_data = train_data.with_transform(transform_batch) # when train_data[:batch_size] HF grab and fed it into tranform_batch(.) | with_transform triggers only to slices

data_loader = DataLoader(
    train_data,
    batch_size=2,
)
# shuffle: how it happens and how do it to (image,label) not destroy?
# num_workers
# collator: a custom collate_fn so your loop yields explicit xb, yb tuples instead of a dictionary? This would allow you to write for xb, yb in data_loader: directly.
# default_collator
for batch in data_loader:
    xb = batch["image"]
    yb = batch["label"]
    print(xb.shape, yb.shape)
    print(batch)
    break