import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_from_disk, load_dataset, Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import os
import sys; sys.path.append(".")
from config import Config
from multiprocessing import Value
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
# redo to def inside def and without wrapper
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
        self.finetune = False
        self._iter_counter = Value('i', -1) # shared int across workers

    def step(self):
        i = self._iter_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, list_of_i1l1_dicts):
        B = len(list_of_i1l1_dicts)
        xbyb_dict = torch.utils.data.default_collate(list_of_i1l1_dicts)
        xb = xbyb_dict["image"]

        seed = self.step() # seed 0 at the start
        g = torch.Generator().manual_seed(seed)
        


        # do generetor or still manual_seed(0)?: self._itr_counter, def step(self) in src/masks/multiblock.py
        # sys.exit(0)
        
        # src/masks/multiblock.py
        # mask strategy for single gpu. for ddp change the code.
        # torch.randint(self.num_patches)
        context_mask_patch_indices = None
        target_mask4_patch_indices = None

        return_list = [xb, context_mask_patch_indices, target_mask4_patch_indices]
        if self.finetune:
            yb = xbyb_dict["label"]
            return_list.append(yb)
            return *return_list,
        return *return_list,

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
train_data = train_data.with_transform(make_transform)
mask_collator = Mask_collator()

if __name__=="__main__":
    data_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        collate_fn=mask_collator,
        num_workers=cfg.num_workers,
    )
    # shuffle or sampler(<- i need it): how it happens and how do it to (image,label) not destroy?
    # collate_fn in DataLoader is the thing that mentioned in masking strategy ijepa paper page 12 (gemini said).

    for xb, context_indecies, targets_indecies in data_loader:
        print(xb.shape)
        # print(context_indecies, targets_indecies)
        print(xb)
        break