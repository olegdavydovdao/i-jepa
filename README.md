# I-JEPA
Reimplement the I-JEPA paper from scratch.

## Fix a bug in original I-JEPA
The bug is that the context and target masks never select the last row and last column of the patch grid.\
For example: in patch grid (14,14). 13 row and 13 column is always 0 and can't be choseen 1 by code.\
This cause that these patches never used for context or target masks, but still computes.\
In case (14,14) patch grid 14+13=27 patches is always computes by masking strategy and computes in main model but never selected\
13.7% computes are wasted and context_size_scale == (0.85, 1.0) is don't work because always get (13,13) and this static image in context.

Path to my code: Mask_collator class/_sample_block_size function
```python
while h > cfg.height: # > instead of >=
    h -= 1
while w > cfg.width: # > instead of >=
    w -= 1
```
Path to my code: Mask_collator class/_sample_block_mask function
```python
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