from .base import TimeSampler

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs


class LinearAnnealingTimeSampler(TimeSampler):
    @ignore_kwargs
    @dataclass
    class Config(TimeSampler.Config):
        pass

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    def __call__(self, step):
        ratio = step / self.cfg.max_steps  # 0.0 ~ 1.0
        t_curr = int(self.cfg.t_max + (self.cfg.t_min - self.cfg.t_max) * ratio)
        t_curr = max(0, min(999, t_curr))

        return t_curr 
