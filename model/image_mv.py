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

from utils.image_utils import save_tensor, pil_to_torch


class ImageMVModel(BaseModel):
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
        initialization: str = "random"  # random, zero, gray, image
        init_img_path: Optional[str] = None
        channels: int = 3
        batch_size: int = 1

        learning_rate: float = 0.1

        ddim_style_generation: bool = False

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None

    def load(self, path: str) -> None:
        img = Image.open(path).convert("RGB")
        img = pil_to_torch(img).to(self.cfg.device).squeeze(0)

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
        B = self.cfg.batch_size
        H, W = self.cfg.height, self.cfg.width
        if self.cfg.channels == 3:
            print("Detected 3 channels. Assuming RGB-space image.")
            if self.cfg.initialization == "random":
                self.image = torch.rand(B, 3, H, W, device=self.cfg.device)
            elif self.cfg.initialization == "zero":
                self.image = torch.zeros(B, 3, H, W, device=self.cfg.device)
            elif self.cfg.initialization == "gray":
                self.image = torch.full((B, 3, H, W), 0.5, device=self.cfg.device)
            elif self.cfg.initialization == "image":
                self.load(self.cfg.init_img_path)
            else:
                raise ValueError(f"Invalid initialization: {self.cfg.initialization}")
        elif self.cfg.channels == 4:
            print("Detected 4 channels. Assuming latent-space image.")
            if self.cfg.initialization == "random":
                self.image = torch.randn(B, 4, H, W, device=self.cfg.device)
            elif self.cfg.initialization == "zero":
                scaling_factor = int(shared_modules.prior.pipeline.vae_scale_factor)
                self.image = shared_modules.prior.encode_image(
                    torch.zeros(
                        B,
                        4,
                        H * scaling_factor,
                        W * scaling_factor,
                        device=self.cfg.device,
                    )
                )
            elif self.cfg.initialization == "gray":
                scaling_factor = int(shared_modules.prior.pipeline.vae_scale_factor)
                self.image = shared_modules.prior.encode_image(
                    torch.full(
                        (B, 4, H * scaling_factor, W * scaling_factor),
                        0.5,
                        device=self.cfg.device,
                    )
                )
            elif self.cfg.initialization == "zero_latent":
                self.image = torch.zeros(B, 4, H, W, device=self.cfg.device)
            elif self.cfg.initialization == "image":
                self.load(self.cfg.init_img_path)
            else:
                raise ValueError(f"Invalid initialization: {self.cfg.initialization}")
        else:
            raise ValueError(f"Channels must be 3 or 4, got {self.cfg.channels}")

        self.image = torch.nn.Parameter(self.image)

        if not self.cfg.ddim_style_generation:
            self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)
        else:
            self.optimizer = torch.optim.SGD([self.image], lr=self.cfg.learning_rate)

    def render(self, camera) -> torch.Tensor:
        img_resized = F.interpolate(
            self.image,
            size=(camera["height"], camera["width"]),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "image": img_resized,
            "alpha": torch.ones(
                self.cfg.batch_size, 1, self.cfg.height, self.cfg.width, device=self.cfg.device
            ),
        }

    def optimize(self, step: int) -> None:
        self.schedule_lr(step)
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def schedule_lr(self, step):
        """
        Adjust the learning rate (optional).
        """
        if not self.cfg.ddim_style_generation:
            return
        timestep = shared_modules.sampler.sample_timestep(step)
        for _ in range(10):
            assert timestep == shared_modules.sampler.sample_timestep(step), "timestep sampling must be deterministic for DDIM-style generation."
        
        alphas = shared_modules.prior.scheduler.alphas_cumprod
        lr = 1 / (alphas[timestep] * (1 - alphas[timestep]))**0.5

        for group in self.optimizer.param_groups:
            group['lr'] = lr