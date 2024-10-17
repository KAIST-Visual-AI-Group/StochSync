from random import choice, choices, randint
from typing import Tuple, List
from dataclasses import dataclass
from dataclasses import field

import torch
# from torchvision import transforms
from torchvision.io import read_video, write_video

from utils.extra_utils import ignore_kwargs
from .base import InfiniteDataset


class VideoDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        interval: int = 5
        video_path: str = "data/video.mp4"
        max_frames: int = -1

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)

        self.ofs = 0
        frames, _, _ = read_video(self.cfg.video_path)
        self.num_frames = frames.shape[0]
        if self.cfg.max_frames > 0:
            self.num_frames = min(self.num_frames, self.cfg.max_frames)
        del frames

    def generate_sample(self):
        frame_idx = torch.arange(0, self.num_frames, self.cfg.interval) + self.ofs
        if frame_idx[-1] >= self.num_frames:
            frame_idx = frame_idx[:-1]
        # to list
        frame_idx = frame_idx.tolist()
        if frame_idx[-1] >= self.num_frames:
            frame_idx = frame_idx[:-1]

        self.ofs = (self.ofs + 2) % self.cfg.interval
        return {
            "num": len(frame_idx),
            "height": self.cfg.height,
            "width": self.cfg.width,
            "frame_idx": frame_idx,
        }