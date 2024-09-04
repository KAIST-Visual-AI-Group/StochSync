from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs

class TimeSampler(ABC):
    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 10000
        t_min: int = 20
        t_max: int = 980
    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    @abstractmethod
    def __call__(self, step):
        # Return timestep 
        pass
