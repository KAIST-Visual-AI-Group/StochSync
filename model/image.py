import os
import sys
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

from .base import BaseModel

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
import shared_modules

from k_utils.image_utils import save_tensor


class ImageModel(BaseModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 10000
        device: str = "cuda"
        width: int = 512
        height: int = 512
        initialization: str = "gray"  # random, zero
        channels: int = 3

        learning_rate: float = 0.1

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None

    def load(self, path: str) -> None:
        img = Image.open(path).convert("RGB")
        img = np.array(img) / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        if self.cfg.channels == 3:
            self.image = torch.nn.Parameter(img)
        elif self.cfg.channels == 4:
            latent = shared_modules.prior.encode_image(img.unsqueeze(0)).squeeze(0)
            self.image = torch.nn.Parameter(latent)
        else:
            raise ValueError(f"Channels must be 3 or 4, got {self.cfg.channels}")

    def save(self, path: str) -> None:
        if self.cfg.channels == 3:
            save_tensor(self.image, path)
        elif self.cfg.channels == 4:
            img = shared_modules.prior.decode_latent(self.image.unsqueeze(0)).squeeze(0)
            save_tensor(img, path)
        else:
            raise ValueError(f"Channels must be 3 or 4, got {self.cfg.channels}")

    def prepare_optimization(self) -> None:
        if self.cfg.channels == 3:
            print("Detected 3 channels. Assuming RGB-space image.")
            if self.cfg.initialization == "random":
                self.image = torch.rand(
                    3, self.cfg.height, self.cfg.width, device=self.cfg.device
                )
            elif self.cfg.initialization == "zero":
                self.image = torch.zeros(
                    3, self.cfg.height, self.cfg.width, device=self.cfg.device
                )
            elif self.cfg.initialization == "gray":
                self.image = torch.full(
                    (3, self.cfg.height, self.cfg.width), 0.5, device=self.cfg.device
                )
            else:
                raise ValueError(f"Invalid initialization: {self.cfg.initialization}")
        elif self.cfg.channels == 4:
            print("Detected 4 channels. Assuming latent-space image.")
            if self.cfg.initialization == "random":
                self.image = torch.randn(
                    4, self.cfg.height, self.cfg.width, device=self.cfg.device
                )
            elif self.cfg.initialization == "zero":
                scaling_factor = int(shared_modules.prior.pipeline.vae_scale_factor)
                self.image = torch.zeros(
                    3,
                    self.cfg.height * scaling_factor,
                    self.cfg.width * scaling_factor,
                    device=self.cfg.device,
                )
                self.image = shared_modules.prior.encode_image(self.image)
            elif self.cfg.initialization == "gray":
                scaling_factor = int(shared_modules.prior.pipeline.vae_scale_factor)
                self.image = torch.full(
                    (
                        3,
                        self.cfg.height * scaling_factor,
                        self.cfg.width * scaling_factor,
                    ),
                    0.5,
                    device=self.cfg.device,
                )
                self.image = shared_modules.prior.encode_image(self.image)
            elif self.cfg.initialization == "zero_latent":
                self.image = torch.zeros(
                    4, self.cfg.height, self.cfg.width, device=self.cfg.device
                )
            else:
                raise ValueError(f"Invalid initialization: {self.cfg.initialization}")
        else:
            raise ValueError(f"Channels must be 3 or 4, got {self.cfg.channels}")

        self.image = torch.nn.Parameter(self.image)
        self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)

    def render(self, camera) -> torch.Tensor:
        tf = camera.get("transforms", lambda x: x)
        img = tf(self.image.unsqueeze(0))

        img_resized = F.interpolate(
            img,
            size=(camera["height"], camera["width"]),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "image": img_resized,
            "alpha": torch.ones(
                1, 1, self.cfg.height, self.cfg.width, device=self.cfg.device
            ),
        }

    def optimize(self, step: int) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()