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
        return {"height": self.cfg.height, "width": self.cfg.width}


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
            "num": self.cfg.batch_size,
            "height": self.cfg.height,
            "width": self.cfg.width,
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
            "num": self.cfg.batch_size,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "transforms": tfs,
        }


class ImageWideRandomDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        xscale: int = 4
        yscale: int = 1
        batch_size: int = 1

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use Pytorch API
        yoffsets = torch.randint(
            0, self.cfg.height * (self.cfg.yscale - 1) + 1, (self.cfg.batch_size,)
        )
        xoffsets = torch.randint(
            0, self.cfg.width * (self.cfg.xscale - 1) + 1, (self.cfg.batch_size,)
        )
        return {
            "num": self.cfg.batch_size,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "yoffsets": yoffsets,
            "xoffsets": xoffsets,
        }


class ImageWideDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        xscale: int = 4
        yscale: int = 1
        batch_size: int = 1

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.cfg.yscale == 1, "ImageWideDataset only supports yscale=1"
        xoffsets = torch.linspace(
            0, self.cfg.width * (self.cfg.xscale - 1), self.cfg.batch_size, dtype=torch.long
        )
        yoffsets = torch.zeros(self.cfg.batch_size, dtype=torch.long)

        return {
            "num": self.cfg.batch_size,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "yoffsets": yoffsets,
            "xoffsets": xoffsets,
        }
