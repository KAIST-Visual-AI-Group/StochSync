from dataclasses import dataclass
from typing import Optional

import torch
from .image import ImageModel

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
import shared_modules as sm

from utils.print_utils import print_with_box, print_warning


class ImageWideModel(ImageModel):
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
        xscale: float = 4
        yscale: float = 1
        
        latent_scale: int = 8
        learning_rate: float = 0.1

    def __init__(self, cfg={}):
        super().__init__()
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
            )
        )
        # self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)
        # self.optimizer = torch.optim.SGD([self.image], lr=self.cfg.learning_rate)
        # momentum optimizer
        self.optimizer = torch.optim.SGD([self.image], lr=self.cfg.learning_rate, momentum=0.9)

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
            
        noise_map = torch.rand_like(self.image)
        
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
        # if self.momentum is None:
        #     self.momentum = (img_new - self.image)
        # else:
        #     self.momentum = 0.9 * self.momentum + (img_new - self.image)
        # lr = 0.02 + 0.05*(step/300)
        # self.image = self.image + self.momentum * lr
        # self.image = 0.9 * self.image + 0.1 * img_new
        
        
    def compute_reproj_error(self, target, camera):
        import torch.nn.functional as F
        
        if self.image.shape[0] == 3:
            target = sm.prior.decode_latent_if_needed(target)
        elif self.image.shape[0] == 4:
            target = sm.prior.encode_image_if_needed(target)
        
        num_cameras = camera["num"]
        yoffsets, xoffsets = camera["yoffsets"], camera["xoffsets"]
        height, width = camera["height"], camera["width"]

        # 1. Unprojected image 
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
        
        # 2. Project image 
        img_cropped = []
        for i in range(num_cameras):
            img_cropped.append(
                img_new[
                    :,
                    yoffsets[i] : yoffsets[i] + height,
                    xoffsets[i] : xoffsets[i] + width,
                ]
            )
        img_cropped = torch.stack(img_cropped, dim=0)
        
        reproj_error = F.mse_loss(
            target, img_cropped, reduction="mean"
        ) / camera["num"]
        
        return reproj_error
