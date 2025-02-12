from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from ..utils.image_utils import save_tensor, pil_to_torch
from ..utils.print_utils import print_info
from ..utils.extra_utils import ignore_kwargs
from .. import shared_modules

from .base import BaseModel
from .image import ImageModel

class ImageInpaintingModel(ImageModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config(ImageModel.Config):
        gt_image_path: Optional[str] = None

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.gt_image = None
        self.gt_mask = None

    def prepare_optimization(self) -> None:
        self.image = torch.nn.Parameter(
            self.initialize_image(
                self.cfg.channels,
                self.cfg.height,
                self.cfg.width,
                self.cfg.initialization,
                self.cfg.init_img_path,
            )
        )

        if self.cfg.gt_image_path is not None:
            self.gt_image = pil_to_torch(Image.open(self.cfg.gt_image_path)).squeeze(0).to(self.cfg.device)
            self.gt_mask = (self.gt_image.sum(0) != 0).float()
            if self.cfg.channels == 3:
                self.gt_image = shared_modules.prior.decode_latent_if_needed(self.gt_image)
            elif self.cfg.channels == 4:
                self.gt_image = shared_modules.prior.encode_image_if_needed(self.gt_image)
            
            # interpolate the mask to the same size as the image (nearest neighbor)
            self.gt_mask = F.interpolate(self.gt_mask.unsqueeze(0).unsqueeze(0), (self.cfg.height, self.cfg.width), mode="nearest").squeeze(0)
            print(self.gt_image.shape, self.gt_mask.shape)
        
        self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)

    def closed_form_optimize(self, step, camera, target):
        if self.image.shape[0] == 3:
            target = shared_modules.prior.decode_latent_if_needed(target)
        elif self.image.shape[0] == 4:
            target = shared_modules.prior.encode_image_if_needed(target)

        assert target.shape[0] == 1, "Target must have batch size 1"
        self.image = target.squeeze(0)

        self.image = self.gt_image * self.gt_mask + self.image * (1 - self.gt_mask)