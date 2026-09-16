import torch
from datasets import load_dataset, Dataset
from torchvision.transforms import v2
import sys; sys.path.append(".")
from multiprocessing import Value
import math
from logging import getLogger
logger = getLogger()

# --------------------------------------------------------------------------------

# Install small subset of Imagenet1k
def install_data_folder_tiny(cfg, split):
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

def install_all_imagenet1k(split, cfg):
    # Note: code slightly will change in calling stage due without .save_to_disk()
    pass

# n_rows with raw images -> n_rows with 224x224 crop tensor for each image independently
class Make_transform():
    def __init__(self, cfg):
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
# My mask strategy is fixed and specialized by config. To general case masks need add safety code to do experiments.
class Mask_collator():
    def __init__(self, cfg, finetune=False):
        self.finetune = finetune
        self._iter_counter = Value('i', -1) # shared int across workers
        self.cfg = cfg

    # change generator seed for every collate_fn call, each worker increase counter.
    def step(self):
        i = self._iter_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    # sample 1 aspect_ratio and 1 mask_scale to 1 batch with generator seed
    def _sample_block_size(self, g, mask_scale_range, aspect_ratio_range):
        _rand = torch.rand(1, generator=g).item()
        min_s, max_s = mask_scale_range
        mask_scale = min_s + _rand * (max_s-min_s)
        max_keep = int(mask_scale * self.cfg.num_patches)
        min_ar, max_ar = aspect_ratio_range
        aspect_ratio = min_ar + _rand * (max_ar-min_ar)
        # aspect_ratio = h/w | max_keep = h*w
        h = round(math.sqrt(max_keep * aspect_ratio))
        w = round(math.sqrt(max_keep / aspect_ratio))
        while h > self.cfg.height:
            h -= 1
        while w > self.cfg.width:
            w -= 1
        return (h,w) # max_h == cfg.height

    def _sample_block_mask(self, mask_size, masks_t_inv=None):
        # with targets masks_t_inv==None | with context masks_t_inv==not None
        def context_rm_overlap_targets(mask):
            N = len(masks_t_inv)
            for k in range(N):
                mask *= masks_t_inv[k]
            return mask

        h,w = mask_size
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # sample top-left corner of the mask block
            top = torch.randint(0, 1+self.cfg.height - h, (1,))
            left = torch.randint(0, 1+self.cfg.width - w, (1,))
            # from top-left corner draw the mask
            mask = torch.zeros((self.cfg.height, self.cfg.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            if masks_t_inv is not None: # for context
                mask = context_rm_overlap_targets(mask)
            mask_indices = torch.nonzero(mask.flatten())
            valid_mask = len(mask_indices) >= self.cfg.min_mask_num_patches
            if not valid_mask: # in my case: it only for context masks
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    if h + 1 <= self.cfg.height:
                        h += 1
                    if w + 1 <= self.cfg.width:
                        w += 1
                    logger.warning(f"Mask is too small, tries:{tries} | new (h,w) == {h,w}")

        mask_indices = mask_indices.squeeze()
        mask_inverse = None
        if masks_t_inv is None: # for targets
            mask_inverse = torch.ones((self.cfg.height, self.cfg.width), dtype=torch.int32)
            mask_inverse[top:top+h, left:left+w] = 0
        return mask_indices, mask_inverse

    def __call__(self, list_of_i1l1_dicts):
        # get xb tensor (B,C,H,W) in my case (B,3,224,224)
        B = len(list_of_i1l1_dicts)
        xbyb_dict = torch.utils.data.default_collate(list_of_i1l1_dicts)
        xb = xbyb_dict["image"]

        # seed to mask shape reproducibility
        seed = self.step() # seed 0 at the start
        g = torch.Generator().manual_seed(seed)

        # get masks sizes shared to 1 Batch with generator seed
        target_size = self._sample_block_size(g, self.cfg.target_mask_scale_range, self.cfg.target_aspect_ratio_range)
        context_size = self._sample_block_size(g, self.cfg.context_mask_scale_range, self.cfg.context_aspect_ratio_range)

        # get masks: locations without generator seed
        collated_t_idxs, collated_c_idxs = [],[]
        min_keep_target = self.cfg.num_patches
        min_keep_context= self.cfg.num_patches
        for _ in range(B): # loop over B images
            # get M target masks for each image. here M==4
            masks_t_idxs, masks_t_inv = [], []
            for _ in range(self.cfg.num_target_masks):
                mask_t_indicies, mask_t_inverse = self._sample_block_mask(target_size)
                masks_t_idxs.append(mask_t_indicies)
                masks_t_inv.append(mask_t_inverse)
                min_keep_target = min(min_keep_target, len(mask_t_indicies))
            collated_t_idxs.append(masks_t_idxs)

            # get context mask for each image
            if self.cfg.allow_overlap:
                masks_t_inv = None
            masks_c_idxs = []
            for _ in range(self.cfg.num_context_masks): # 1
                mask_c_indicies, _ = self._sample_block_mask(context_size, masks_t_inv=masks_t_inv)
                masks_c_idxs.append(mask_c_indicies)
                min_keep_context = min(min_keep_context, len(mask_c_indicies))
            collated_c_idxs.append(masks_c_idxs)

        # list of list of 4 tensors. Restrict num of patches to be equal over a batch. Cause: effective GPU batch matrix multiply.
        collated_t_idxs = [[t_mask[:min_keep_target] for t_mask in list_n_ts] for list_n_ts in collated_t_idxs] # in my case is always equal but i want this shield.
        collated_t_idxs = torch.utils.data.default_collate(collated_t_idxs)
        collated_c_idxs = [[c_mask[:min_keep_context] for c_mask in list_n_c] for list_n_c in collated_c_idxs]
        collated_c_idxs = torch.utils.data.default_collate(collated_c_idxs)

        # return list of elements with respect to stage(pretrain SSL or finetune)
        if self.finetune:
            yb = xbyb_dict["label"]
            return xb, yb
        return xb, collated_c_idxs, collated_t_idxs
