from random import choice, choices
from typing import Tuple, List
from dataclasses import dataclass, field

import torch

from .base import InfiniteDataset
from ..utils.extra_utils import ignore_kwargs


class VideoDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        batch_size: int = 1
        n_frames: int = 10
        start_frames: Tuple[int] = (0,)
        end_frames: Tuple[int] = (10,)

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.cnt = 0
        assert len(self.cfg.start_frames) == len(self.cfg.end_frames), "Length of start_frames and end_frames must be the same"

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        start_frames = self.cfg.start_frames
        end_frames = self.cfg.end_frames

        B = self.cfg.batch_size
        chunk = B * (self.cnt % ((len(start_frames) + B - 1) // B))
        start_frames = start_frames[chunk:chunk+B]
        end_frames = end_frames[chunk:chunk+B]
        self.cnt += 1
        return {
            "num": B,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "start_frames": start_frames,
            "end_frames": end_frames,
        }


class VideoDataset(ImageDataset):
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