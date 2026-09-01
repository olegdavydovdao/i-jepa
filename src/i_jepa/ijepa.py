import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_from_disk

# hyperparameters
torch.manual_seed(52)

# ImageNet data prepare, data loader
train_data = load_from_disk('imagenet1k_1000rows')
print(train_data[:10])
# images into 224x224px + this happens before installation and save to imagenet1k_1000rows