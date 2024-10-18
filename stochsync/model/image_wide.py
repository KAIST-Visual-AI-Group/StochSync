from dataclasses import dataclass
from typing import Optional

import torch

from ..utils.extra_utils import ignore_kwargs
from ..utils.print_utils import print_with_box, print_warning
from .. import shared_modules as sm

from .image import ImageModel


class ImageWideModel(ImageModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config(ImageModel.Config):
        xscale: float = 4
        yscale: float = 1

    def __init__(self, cfg={}):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None
        self.momentum = None

    def prepare_optimization(self) -> None:
        self.image = torch.nn.Parameter(
            self.initialize_image(
                self.cfg.channels,
                int(self.cfg.height * self.cfg.yscale),
                int(self.cfg.width * self.cfg.xscale),
                self.cfg.initialization,
                self.cfg.init_img_path,
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
        
        
    def get_noise(self, camera) -> torch.Tensor:
        if self.cfg.initialization != "random":
            print_warning("get_noise called for non-random initialization.")
            
        noise_map = torch.randn_like(self.image)
        
        num_cameras = camera["num"]
        yoffsets, xoffsets = camera["yoffsets"], camera["xoffsets"]
        height, width = camera["height"], camera["width"]
        
        xt_cropped = []
        for i in range(num_cameras):
            xt_cropped.append(
                noise_map[
                    :,
                    yoffsets[i] : yoffsets[i] + height,
                    xoffsets[i] : xoffsets[i] + width,
                ]
            )
            
        xts = torch.stack(xt_cropped, dim=0)
        
        return xts 
        

    def closed_form_optimize(self, step, camera, target):
        if self.image.shape[0] == 3:
            target = sm.prior.decode_latent_if_needed(target)
        elif self.image.shape[0] == 4:
            target = sm.prior.encode_image_if_needed(target)
        
        num_cameras = camera["num"]
        yoffsets, xoffsets = camera["yoffsets"], camera["xoffsets"]
        height, width = camera["height"], camera["width"]

        img_new = torch.zeros_like(self.image)
        img_cnt = torch.zeros_like(self.image, dtype=torch.long)
        for i in range(num_cameras):
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

        self.image = img_new