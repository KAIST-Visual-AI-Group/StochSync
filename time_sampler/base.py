from abc import ABC, abstractmethod

import random
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
from math import pi, sin

class TimeSampler(ABC):
    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 10000
        t_min: int = 20
        t_max: int = 980
        batch_size: int = 10 # batch size
    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    @abstractmethod
    def __call__(self, step):
        # Return timestep 
        pass

class RepeatingTimeSampler(TimeSampler):
    @ignore_kwargs
    @dataclass
    class Config(TimeSampler.Config):
        t_repeat: int = 900

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    def __call__(self, step):
        if step == 0:
            return self.cfg.t_max
        return self.cfg.t_repeat
    
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

class GoodTimeSampler(TimeSampler):
    @ignore_kwargs
    @dataclass
    class Config(TimeSampler.Config):
        t_min: int = 250
        t_max: int = 900

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    def __call__(self, step):
        ratio = step / self.cfg.max_steps  # 0.0 ~ 1.0

        y= (ratio + sin(2*pi*ratio)/(2*pi))**1.2
        t_curr = int(self.cfg.t_max + (self.cfg.t_min - self.cfg.t_max) * y)
        t_curr = max(0, min(999, t_curr))

        return t_curr

class SDSTimeSampler(TimeSampler):
    @ignore_kwargs
    @dataclass
    class Config(TimeSampler.Config):
        pass
    
    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)

    def __call__(self, step):
        t_curr = random.randint(self.cfg.t_min, self.cfg.t_max)
        return t_curr