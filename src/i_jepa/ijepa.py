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
        # if else with install_all_data to load it later in code because load_from_disk is not working without .save_to_disk
        # train_set = load_dataset('ILSVRC/imagenet-1k', split='train', cache_dir=folder_name) # local_files_only=True
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

def mask_collator(list_of_i1l1_dicts): # HF Dataset.__getitem__ change the HF type of data
    list_images = [il_dict["image"] for il_dict in list_of_i1l1_dicts]
    xb = torch.stack(list_images)
    return xb

# ImageNet data prepare
train_data = load_from_disk(folder_name)
transform = make_transform()
# when train_data[:batch_size] HF grab and fed it into tranform_batch(.) | with_transform triggers only to slices
train_data = train_data.with_transform(transform_batch)

data_loader = DataLoader(
    train_data,
    batch_size=2,
    collate_fn=mask_collator
)
# shuffle or sampler(<- i need it): how it happens and how do it to (image,label) not destroy?
# num_workers
# collator: a custom collate_fn so your loop yields explicit xb, yb tuples instead of a dictionary? This would allow you to write for xb, yb in data_loader: directly.
# default_collator, collate_fn in DataLoader is the thing that mentioned in masking strategy ijepa paper page 12 (gemini said).
for xb in data_loader:
    print(xb.shape)
    print(xb)
    break