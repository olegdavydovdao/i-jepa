import sys
import math

class LRScheduler: # Warmup and Cosine
    def __init__(self, cfg, i_per_ep):
        self._step = 0
        self.warmup_steps = int(cfg.warmup_over_epochs*cfg.num_epochs*i_per_ep)
        self.T_max = int(cfg.num_epochs*i_per_ep) - self.warmup_steps
        self.start_lr = cfg.start_lr
        self.max_lr = cfg.max_lr
        self.final_lr = cfg.final_lr
        assert self.warmup_steps != 0 and self.T_max !=0, '(warmup steps or T_max) == 0'

    def step(self):
        if self._step < self.warmup_steps:
            ratio = self._step / self.warmup_steps
            # linear from start_lr to max_lr
            new_lr = self.start_lr + ratio * (self.max_lr - self.start_lr)
        else:
            ratio = (self._step - self.warmup_steps) / self.T_max
            new_lr = self.final_lr + (self.max_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi*ratio))
            new_lr = max(new_lr, self.final_lr)
        self._step += 1
        return new_lr