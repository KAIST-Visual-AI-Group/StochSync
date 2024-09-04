from .base import TimeSampler

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs


class SDSTimeSampler(TimeSampler):
    @ignore_kwargs
    @dataclass
    class Config(TimeSampler.Config):
        pass
    
    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    def __call__(self, step):
        t_curr = torch.randint(self.cfg.t_min, self.cfg.t_max+1)
        return t_curr 
        
