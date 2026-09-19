# I-JEPA
Reimplement the I-JEPA paper from scratch.

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

## Citations

I-JEPA original

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```

ImageNet-1k

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