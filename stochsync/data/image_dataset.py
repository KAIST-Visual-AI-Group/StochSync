from random import choice, choices
from typing import Tuple, List
from dataclasses import dataclass, field

import torch

from .base import InfiniteDataset
from ..utils.extra_utils import ignore_kwargs


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


class ImageWideDataset(ImageDataset):
    @ignore_kwargs
    @dataclass
    class Config(ImageDataset.Config):
        xscale: int = 1
        yscale: int = 1

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            self.cfg.xscale == 1 or self.cfg.yscale == 1
        ), "ImageWideDataset only supports xscale=1 or yscale=1"
        if self.cfg.xscale == 1:
            end = self.cfg.height * (self.cfg.yscale - 1)
            yoffsets = torch.linspace(0, end, self.cfg.batch_size, dtype=torch.long)
            xoffsets = torch.zeros_like(yoffsets)
        else:
            end = self.cfg.width * (self.cfg.xscale - 1)
            xoffsets = torch.linspace(0, end, self.cfg.batch_size, dtype=torch.long)
            yoffsets = torch.zeros_like(xoffsets)

        return {
            "num": self.cfg.batch_size,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "yoffsets": yoffsets,
            "xoffsets": xoffsets,
        }


class RandomImageWideDataset(ImageWideDataset):
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.cfg.xscale == 1 or self.cfg.yscale == 1, "ImageWideDataset only supports xscale=1 or yscale=1"
        if self.cfg.xscale == 1:
            end = self.cfg.height * (self.cfg.yscale - 1)
            yoffsets = torch.randint(0, end + 1, (self.cfg.batch_size,), dtype=torch.long)
            xoffsets = torch.zeros_like(yoffsets)
        else:
            end = self.cfg.width * (self.cfg.xscale - 1)
            xoffsets = torch.randint(0, end + 1, (self.cfg.batch_size,), dtype=torch.long)
            yoffsets = torch.zeros_like(xoffsets)

        return {
            "num": self.cfg.batch_size,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "yoffsets": yoffsets,
            "xoffsets": xoffsets,
        }


# ===================================================
# =========== Debugging dataset =====================
# ===================================================


class AlternateImageWideDataset(ImageWideDataset):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.flag = False

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.cfg.yscale == 1, "ImageWideDataset only supports yscale=1"
        xoffsets = torch.linspace(
            0,
            self.cfg.width * (self.cfg.xscale - 1),
            self.cfg.batch_size,
            dtype=torch.long,
        )
        yoffsets = torch.zeros(self.cfg.batch_size, dtype=torch.long)

        if self.flag:
            xoffsets = xoffsets[1::2]
            yoffsets = yoffsets[1::2]
        else:
            xoffsets = xoffsets[::2]
            yoffsets = yoffsets[::2]
        self.flag = not self.flag

        return {
            "num": len(xoffsets),
            "height": self.cfg.height,
            "width": self.cfg.width,
            "yoffsets": yoffsets,
            "xoffsets": xoffsets,
        }


class SixViewNoOverlapCameraDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        xscale: int = 6
        yscale: int = 1
        batch_size: int = 1

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            self.cfg.yscale == 1
        ), "TwoViewNoOverlapCameraDataset only supports yscale=1"

        xoffsets = torch.linspace(
            0,
            self.cfg.width * (self.cfg.xscale - 1),
            self.cfg.batch_size,
            dtype=torch.long,
        )
        print("xoffsets", xoffsets)

        yoffsets = torch.zeros(self.cfg.batch_size, dtype=torch.long)

        return {
            "num": self.cfg.batch_size,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "yoffsets": yoffsets,
            "xoffsets": xoffsets,
        }
