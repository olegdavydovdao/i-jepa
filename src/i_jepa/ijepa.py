import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_from_disk
from torchvision.transforms import v2

# hyperparameters
torch.manual_seed(0)

# raw imagenet1k image -> 224x224 crop tensor
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
    return transform # out = transform(img)

# ImageNet data prepare, data loader
train_data = load_from_disk('imagenet1k_1000rows')
img = train_data[0]["image"]
transform = make_transform()
out = transform(img)
print(img)
print(out.shape, out.dtype)