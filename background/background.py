import torch
from dataclasses import dataclass

from .base import BaseBackground
from utils.extra_utils import ignore_kwargs
import shared_modules

# Using weak_lru to avoid memory leaks in class methods
from utils.extra_utils import weak_lru


class SolidBackground(BaseBackground):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        device: str = "cuda"
        rgb: tuple = (1, 1, 1)

    def __init__(self, cfg) -> None:
        self.cfg = self.Config(**cfg)
        self.background = torch.tensor(self.cfg.rgb, device=self.cfg.device).view(
            1, -1, 1, 1
        )

    def __call__(self, camera) -> torch.Tensor:
        return self.background


class LatentSolidBackground(BaseBackground):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        device: str = "cuda"
        rgb: tuple = (1, 1, 1)

    def __init__(self, cfg) -> None:
        self.cfg = self.Config(**cfg)
        self.background = (
            torch.tensor(self.cfg.rgb, device=self.cfg.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, self.cfg.height * 8, self.cfg.width * 8)
        )

    @weak_lru(maxsize=1)
    def __call__(self, camera) -> torch.Tensor:
        # encode and return
        return shared_modules.prior.encode_image(self.background)


class RandomSolidBackground(BaseBackground):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        mode: str = "hsv"
        device: str = "cuda"
        rgb_range: tuple = (0, 1)
        gray_range: tuple = (0, 1)
        hue_range: tuple = (0, 1)

    def __init__(self, cfg) -> None:
        self.cfg = self.Config(**cfg)

    def __call__(self, camera) -> torch.Tensor:
        if self.cfg.mode == "rgb":
            color = (
                torch.rand(3, device=self.cfg.device)
                * (self.cfg.rgb_range[1] - self.cfg.rgb_range[0])
                + self.cfg.rgb_range[0]
            )
        elif self.cfg.mode == "gray":
            color = (
                torch.rand(1, device=self.cfg.device)
                * (self.cfg.gray_range[1] - self.cfg.gray_range[0])
                + self.cfg.gray_range[0]
            )
        elif self.cfg.mode == "hsv":
            hue = (
                torch.rand(1, device=self.cfg.device)
                * (self.cfg.hue_range[1] - self.cfg.hue_range[0])
                + self.cfg.hue_range[0]
            )
            saturation = torch.rand(1, device=self.cfg.device) * 0.6 + 0.3
            value = torch.rand(1, device=self.cfg.device) * 0.6 + 0.3
            color = torch.cat([hue, saturation, value])
            color = self.hsv_to_rgb(color.unsqueeze(0)).squeeze(0)

        background = (
            color.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, self.cfg.height, self.cfg.width)
        )
        return background

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        i = torch.floor(h * 6).to(torch.int32)
        f = h * 6 - i
        p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
        i = i % 6

        conditions = [
            (i == 0),
            (i == 1),
            (i == 2),
            (i == 3),
            (i == 4),
            (i == 5),
        ]
        rgb = torch.zeros_like(hsv)
        rgb[:, 0] = torch.where(conditions[0], v, rgb[:, 0])
        rgb[:, 1] = torch.where(conditions[0], t, rgb[:, 1])
        rgb[:, 2] = torch.where(conditions[0], p, rgb[:, 2])

        rgb[:, 0] = torch.where(conditions[1], q, rgb[:, 0])
        rgb[:, 1] = torch.where(conditions[1], v, rgb[:, 1])
        rgb[:, 2] = torch.where(conditions[1], p, rgb[:, 2])

        rgb[:, 0] = torch.where(conditions[2], p, rgb[:, 0])
        rgb[:, 1] = torch.where(conditions[2], v, rgb[:, 1])
        rgb[:, 2] = torch.where(conditions[2], t, rgb[:, 2])

        rgb[:, 0] = torch.where(conditions[3], p, rgb[:, 0])
        rgb[:, 1] = torch.where(conditions[3], q, rgb[:, 1])
        rgb[:, 2] = torch.where(conditions[3], v, rgb[:, 2])

        rgb[:, 0] = torch.where(conditions[4], t, rgb[:, 0])
        rgb[:, 1] = torch.where(conditions[4], p, rgb[:, 1])
        rgb[:, 2] = torch.where(conditions[4], v, rgb[:, 2])

        rgb[:, 0] = torch.where(conditions[5], v, rgb[:, 0])
        rgb[:, 1] = torch.where(conditions[5], p, rgb[:, 1])
        rgb[:, 2] = torch.where(conditions[5], q, rgb[:, 2])

        return rgb


class BlackWhiteBackground(BaseBackground):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        white_prob: float = 0.5
        device: str = "cuda"

    def __init__(self, cfg) -> None:
        self.cfg = self.Config(**cfg)

    def __call__(self, camera) -> torch.Tensor:
        color = (torch.rand(1) < self.cfg.white_prob).float().to(self.cfg.device)

        background = (
            color.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, self.cfg.height, self.cfg.width)
        )
        return background

class NeRFBackground(BaseBackground):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        device: str = "cuda"

    def __init__(self, cfg) -> None:
        self.cfg = self.Config(**cfg)

        # Initialize the Directional NeRF model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
            torch.nn.Sigmoid(),
        ).to(self.cfg.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    
    def __call__(self, camera) -> torch.Tensor:
        c2ws, Ks, width, height = (
            camera["c2w"],
            camera["K"],
            camera["width"],
            camera["height"],
        )

        images = []
        for i in range(camera["num"]):
            # Generate rays
            # origins = c2ws[i:i+1, :3, 3] # (1, 3)
            # origins = origins.view(1, 1, 3).expand(height, width, -1)  # (H, W, 3)

            pixel_coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, height - 1, height),
                    torch.linspace(0, width - 1, width),
                    indexing='ij',
                ),
                dim=-1,
            ).flip([2]).to(Ks.device)  # (H, W, 2)
            pixel_coords = pixel_coords.reshape(-1, 2)  # (H*W, 2)

            dirs = Ks[i:i+1].inverse().squeeze() @ torch.cat(
                [
                    pixel_coords,
                    torch.ones(pixel_coords.shape[0], 1).to(pixel_coords.device),
                ],
                dim=1,
            ).T  # (3, 512*512)
            dirs = dirs / torch.norm(dirs, dim=0, keepdim=True)  # (3, H*W)
            # convert to world space
            dirs = c2ws[i:i+1, :3, :3] @ dirs.unsqueeze(0)  # (1, 3, H*W)
            dirs = dirs.squeeze(0).permute(1, 0).view(height, width, 3)  # (H, W, 3)

            # Inference
            rgb = self.model(dirs)
            images.append(rgb)
        images = torch.stack(images, dim=0).permute(0, 3, 1, 2)  # [N, 3, H, W]
        return images
    
    def optimize(self, step):
        self.optimizer.step()
        self.optimizer.zero_grad()