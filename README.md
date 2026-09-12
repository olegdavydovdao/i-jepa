# I-JEPA
Reimplement the I-JEPA paper from scratch.

## Fix a bug in original I-JEPA
For example: in patch grid (14,14).
1) with h >= cfg.height leads to h_max == cfg.height-1 i.e max choosen grid is (13,13)\
this destroy the idea of context_size_scale == (0.85, 1.0), moreover in my case with point 2 it leads to static context mask
2) torch.randint(0, cfg.height - h, (1,)) and point 1 leads to:\
13 row and 13 column of patches never selected by target and context masks.

14+13=27 patches are wasted. 27/196 = 13.77% computes are wasted in mask strategy plus in ViT target computes.\
ViT-context: never get these patches hence not learn.\
ViT-target: computes these patches but never selected as targets, not learn beacuse ViT-target is EMA ViT-context.

```python
# Path to my code: src\i_jepa\data_prepare.py\Mask_collator class\_sample_block_size function \ 69 line
# Path to original code: https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py#L128 \ MaskCollator class\_sample_block_size function \ 67 line
while h > cfg.height: # > instead of >=
    h -= 1
while w > cfg.width: # > instead of >=
    w -= 1
```
```python
# Path to my code: src\i_jepa\data_prepare.py\Mask_collator class\function \ 85 line
# Path to original code: https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py#L128 \ MaskCollator class\_sample_block_mask function \ 89 line
top = torch.randint(0, 1+cfg.height - h, (1,)) # (1+cfg.height - h) instead of (cfg.height - h)
left = torch.randint(0, 1+cfg.width - w, (1,)) # (1+cfg.width - w) instead of (cfg.width - w)
```

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