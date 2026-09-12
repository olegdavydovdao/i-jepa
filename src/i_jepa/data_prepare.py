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
import math
torch.manual_seed(0)

# --------------------------------------------------------------------------------

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

# Collate function and mask strategy
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

    def _sample_block_size(self, g, mask_scale_range, aspect_ratio_range):
        # sample 1 block scale to 1 batch
        _rand = torch.rand(1, generator=g).item()
        min_s, max_s = mask_scale_range
        mask_scale = min_s + _rand * (max_s-min_s)
        max_keep = int(mask_scale * cfg.num_patches)
        # sample 1 block aspect ratio to 1 batch
        min_ar, max_ar = aspect_ratio_range
        aspect_ratio = min_ar + _rand * (max_ar-min_ar)
        # aspect_ratio = h/w | max_keep = h*w
        h = round(math.sqrt(max_keep * aspect_ratio))
        w = round(math.sqrt(max_keep / aspect_ratio))
        while h > cfg.height:
            h -= 1
        while w > cfg.width:
            w -= 1
        return (h,w) # max_h == cfg.height

    def _sample_block_mask(self, mask_size, masks_t_inv=None):
        def context_rm_overlap_targets(mask):
            N = max(len(masks_t_inv), 0)
            for k in range(N):
                mask *= masks_t_inv[k]
            return mask

        h,w = mask_size
        # sample top-left corner of the mask block
        top = torch.randint(0, 1+cfg.height - h, (1,))
        left = torch.randint(0, 1+cfg.width - w, (1,))
        # from top-left corner draw mask
        mask = torch.zeros((cfg.height, cfg.width), dtype=torch.int32)
        mask[top:top+h, left:left+w] = 1
        if masks_t_inv is not None:
            mask = context_rm_overlap_targets(mask)
        mask_indices = torch.nonzero(mask.flatten())
        mask_indices = mask_indices.squeeze()
        mask_inverse = None
        if masks_t_inv is None:
            mask_inverse = torch.ones((cfg.height, cfg.width), dtype=torch.int32)
            mask_inverse[top:top+h, left:left+w] = 0
        return mask_indices, mask_inverse

    def __call__(self, list_of_i1l1_dicts):
        B = len(list_of_i1l1_dicts)
        xbyb_dict = torch.utils.data.default_collate(list_of_i1l1_dicts)
        xb = xbyb_dict["image"]

        seed = self.step() # seed 0 at the start
        g = torch.Generator().manual_seed(seed)

        # get masks sizes 
        target_size = self._sample_block_size(g, cfg.target_mask_scale_range, cfg.target_aspect_ratio_range)
        context_size = self._sample_block_size(g, cfg.context_mask_scale_range, cfg.context_aspect_ratio_range)
        print(f"{target_size=}")
        print(f"{context_size=}")

        # get masks
        collated_t_idxs, collated_c_idxs = [],[]
        min_keep_target = cfg.num_patches
        min_keep_context= cfg.num_patches
        for _ in range(B):
            # target 4 block masks for each image
            masks_t_idxs, masks_t_inv = [], []
            for _ in range(cfg.num_target_masks): # 4
                mask_t_indicies, mask_t_inverse = self._sample_block_mask(target_size)
                masks_t_idxs.append(mask_t_indicies) # list of 4 tensors
                masks_t_inv.append(mask_t_inverse)
                min_keep_target = min(min_keep_target, len(mask_t_indicies))
            collated_t_idxs.append(masks_t_idxs)

            # context block mask for each image
            masks_c_idxs = []
            for _ in range(cfg.num_context_masks): # 1
                mask_c_indicies, _ = self._sample_block_mask(context_size, masks_t_inv=masks_t_inv)
                masks_c_idxs.append(mask_c_indicies)
                min_keep_context = min(min_keep_context, len(mask_c_indicies))
            collated_c_idxs.append(masks_c_idxs)

        # list of list of 4 tensors
        collated_t_idxs = [[t_mask[:min_keep_target] for t_mask in list_n_ts] for list_n_ts in collated_t_idxs]
        collated_t_idxs = torch.utils.data.default_collate(collated_t_idxs)
        collated_c_idxs = [[c_mask[:min_keep_context] for c_mask in list_n_c] for list_n_c in collated_c_idxs]
        collated_c_idxs = torch.utils.data.default_collate(collated_c_idxs)

        return_list = [xb, collated_c_idxs, collated_t_idxs]
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

# --------------------------------------------------------------------------------

if __name__=="__main__":
    data_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        collate_fn=mask_collator,
        num_workers=cfg.num_workers,
    )
    # shuffle or sampler(<- i need it): how it happens and how do it to (image,label) not destroy?

    for xb, context_indecies, targets_indecies in data_loader:
        print(xb.shape)
        print(context_indecies)
        print(targets_indecies)
        # print(xb)
        break