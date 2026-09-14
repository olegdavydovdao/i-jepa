# I-JEPA
Reimplement the I-JEPA paper from scratch.

## Fix a bug in original I-JEPA
Bug in masking strategy:
- Last row and column in patch grid never selected in context and target masks.
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