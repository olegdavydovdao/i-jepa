# I-JEPA
Reimplement the I-JEPA (Image Joint-Embedding Predictive Architecture) paper from scratch in code at pretrain and probe stage.

## Config of a project
Encoder model: ViT-B/16.\
Predictor model: narrow ViT with 6 depth.\
I have small compute capability, but the code is easy to scale up.\
Just change variables in **`config.py`**:
- num_train_rows and num_val_rows: how many rows of data to install (change before 1st run train.py).
- num_workers: how many workers is created inside DataLoader.
- num_epochs and eval_num_epoch: how many epochs to train on.
- use_bfloat16 or use_fp16: which low precision to use.
- batch_size and total_batch_size: reach total_batch_size volume via grad accum steps if not enough vRAM.
- you can change model (ex: Encoder) varialbes too.

## Project structure
i-jepa:
- **`config.py`** — all high level setup variables.
- **`pyproject.toml`** — dependencies.

i-jepa/src/i_jepa:
- **`data_prepare.py`** — install, pre-transform and collate data logic.
- **`models.py`** — encoder, predictor models.
- **`probe.py`** — probe train and test after pre-train stage.
- **`shedulers.py`** — learning rate and weight decay shedulers over time.
- **`train.py`** — i-jepa at pre-training stage.

i-jepa/imagenet1k_tiny:
- **`train`** — installed n rows of train data.
- **`validation`** — installed k rows of val data.\
Note: installing is happens inside train.py file only in 1st time.

i-jepa/checkpoints:
- **`checkpoint_end_pretrain.pt`** — saved state dicts of models and optimizer inside train.py.
- **`checkpoint_end_probe.pt`** — saved state dicts of probe optimizer and model inside probe.py.

## Quick start
**Requirements**: uv, Git, hugging face(HF) account and access to [HF ImageNet1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k),\
minimum 1 Nvidia GPU (tested on Nvidia T4 and Geforce GTX 1050Ti).
```python
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/olegdavydovdao/i-jepa.git
hf auth login
cd i-jepa
uv sync

# DDP run | 1 is num of available GPUs, you can change it.
uv run torchrun --standalone --nproc_per_node=1 src/i_jepa/train.py
uv run torchrun --standalone --nproc_per_node=1 src/i_jepa/probe.py
# default run
uv run python src/i_jepa/train.py
uv run python src/i_jepa/probe.py
```

## Fix bugs in original I-JEPA
### Bug 1 in masking strategy:
Last row and column in patch grid never selected in context and target masks.
```python
# Path to my code: src\i_jepa\data_prepare.py\ 69 and 85 lines
# Path to original code: src\masks\multiblock.py\ 67 and 89 lines

# 69 line:
while h > cfg.height: # > instead of >=
    h -= 1
while w > cfg.width: # > instead of >=
    w -= 1
# 85 line:
top = torch.randint(0, 1+cfg.height - h, (1,)) # (1+cfg.height - h) instead of (cfg.height - h)
left = torch.randint(0, 1+cfg.width - w, (1,)) # (1+cfg.width - w) instead of (cfg.width - w)
```

For example: in patch grid (14,14) with the bug in original I-JEPA code:
1) "while h >= cfg.height" leads to "h_max == cfg.height-1" i.e max choosen grid is (13,13)\
this destroy the idea of context_mask_scale_range == (0.85, 1.0).
2) "torch.randint(0, cfg.height - h, (1,))", with h from point 1 leads to:\
13 row and 13 column of patches (tensor with 14 indicies: from 0 to 13) never selected by target and context masks.\
Context mask is always static.

14+13=27 patches are wasted. 27/196 = 13.77% computes are wasted in mask strategy and ViT-target.\
ViT-context: never get these patches as input hence not learn.\
ViT-target: computes these patches but output never selected as targets, not learn beacuse ViT-target is EMA ViT-context.


### Bug 2 in masking strategy:
**The problem:**\
In original I-JEPA this code prevents infinity loop and useful for experiments without fixed settings to detect which setup of masking strategy how often length of masks are descending to less then threshold, but with fixed I-JEPA standart params I think it is a bug.
```python
N = max(int(len(acceptable_regions)-tries), 0) # 79 line of orig i-jepa multiblock.py
for k in range(N):
    mask *= acceptable_regions[k]
```
But it's introduces a leak in rare cases:\
Context mask can overlap target masks.\
for example tries == 1: context mask not remove the last target mask.\
**My solution:**
```python
N = len(masks_t_inv) # line 78
for k in range(N):
    mask *= masks_t_inv[k]
# if context mask is small.
if h + 1 <= cfg.height: # line 104
    h += 1
if w + 1 <= cfg.width:
    w += 1
```
This code prevents infinity loop and leak information with standart I-JEPA config.\
Shape  (h, w) could be not identical across batch in rare cases,\
but [c_mask[:min_keep_context]] restriction ensures that num of pathces across batch is always identical.\
My code never allow context overlap with targets.

### Naming convention issue in positional embeddings
This non-technical bug (the issue). They do:
```bibtex
# Pseudocode:
emb_h = logical encoding embeddings to width in 2D patch grid
emb_w = logical encoding embeddings to height in 2D patch grid
```
All works fine, even with the logical naming issue.\
Later in code they concat this emb_h and emb_w.\
Positional embeddings technical get both: height and width encoding,\
but not in the order corresponding to the names.

## Citations

I-JEPA original:
- [GitHub Code](https://github.com/facebookresearch/ijepa/tree/main)
- [Paper](https://arxiv.org/pdf/2301.08243)

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```

ImageNet-1k:
- [HF dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

```bibtex
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
```