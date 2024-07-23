from random import choice, choices
from typing import Tuple, List
from dataclasses import dataclass
from dataclasses import field

import torch
from torchvision import transforms

from utils.extra_utils import ignore_kwargs
from .base import InfiniteDataset


class ImageDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        batch_size: int = 1

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return {"width": self.cfg.width, "height": self.cfg.height}


class RotateImageDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        angles: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
        batch_size: int = 1

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        assert all([angle % 90 == 0 for angle in self.cfg.angles])

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = choices(self.cfg.angles, k=self.cfg.batch_size)
        tfs = [transforms.RandomRotation((angle, angle)) for angle in angles]
        return {
            "batch_size": self.cfg.batch_size,
            "width": self.cfg.width,
            "height": self.cfg.height,
            "transforms": tfs,
        }


class RotateBatchImageDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        angles: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
        batch_size: int = 1

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        assert all([angle % 90 == 0 for angle in self.cfg.angles])
        assert len(self.cfg.angles) == self.cfg.batch_size

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = self.cfg.angles
        tfs = [transforms.RandomRotation((angle, angle)) for angle in angles]
        return {
            "batch_size": self.cfg.batch_size,
            "width": self.cfg.width,
            "height": self.cfg.height,
            "transforms": tfs,
        }
