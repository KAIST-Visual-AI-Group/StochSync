from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from .image import ImageModel

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
import shared_modules


class ImageLayerModel(ImageModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config(ImageModel.Config):
        channels: int = 3
        width: int = 512
        height: int = 512
        initialization: str = "random"  # random, zero, gray, image
        init_img_path: Optional[str] = None
        bbox: Tuple[float, float, float, float] = (0.1, 0.25, 0.9, 0.75)
        bg_prompt: str = "An interior of a room with a window."

        learning_rate: float = 0.1

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None

    def prepare_optimization(self) -> None:
        latent_w = self.cfg.width
        latent_h = self.cfg.height
        if self.cfg.channels == 3:
            latent_w //= 8
            latent_h //= 8
        noise = torch.randn(
            1,
            4,
            latent_h,
            latent_w,
            device=self.cfg.device,
        )
        data = shared_modules.dataset.generate_sample()
        self.bg = shared_modules.prior.ddim_loop(data, noise, 999, 0, text_prompt=self.cfg.bg_prompt)
        self.bg = shared_modules.prior.decode_latent(self.bg).squeeze(0)
        self.image = torch.nn.Parameter(
            self.initialize_image(
                self.cfg.channels,
                self.cfg.height * self.cfg.yscale,
                self.cfg.width * self.cfg.xscale,
                self.cfg.initialization,
            )
        )
        self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)

    def render(self, camera) -> torch.Tensor:
        num_cameras = camera["num"]
        yoffsets, xoffsets = camera["yoffsets"], camera["xoffsets"]
        height, width = camera["height"], camera["width"]

        img_cropped = []
        for i in range(num_cameras):
            img_cropped.append(
                self.image[
                    :,
                    yoffsets[i] : yoffsets[i] + height,
                    xoffsets[i] : xoffsets[i] + width,
                ]
            )
        img_cropped = torch.stack(img_cropped, dim=0)

        return {
            "image": img_cropped,
            "alpha": torch.ones(num_cameras, 1, height, width, device=self.cfg.device),
        }

    def closed_form_optimize(self, step, camera, target):
        num_cameras = camera["num"]
        yoffsets, xoffsets = camera["yoffsets"], camera["xoffsets"]
        height, width = camera["height"], camera["width"]

        img_new = torch.zeros_like(self.image)
        img_cnt = torch.zeros_like(self.image, dtype=torch.long)
        for i in range(num_cameras):
            # print(img_new.shape, yoffsets[i], xoffsets[i], target.shape)
            img_new[
                :,
                yoffsets[i] : yoffsets[i] + height,
                xoffsets[i] : xoffsets[i] + width,
            ] += target[i]
            img_cnt[
                :,
                yoffsets[i] : yoffsets[i] + height,
                xoffsets[i] : xoffsets[i] + width,
            ] += 1
        
        img_new = img_new / (img_cnt + 1e-6)
        img_new[img_cnt == 0] = self.image[img_cnt == 0]
        self.image.data = img_new