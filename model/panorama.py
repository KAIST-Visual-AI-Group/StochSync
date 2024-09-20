from dataclasses import dataclass
from typing import Dict, Optional, List

import os 
import torch
import math

from .image import ImageModel

import shared_modules
from torch.optim.lr_scheduler import LambdaLR
from utils.extra_utils import ignore_kwargs
from utils.panorama_utils import pano_to_pers_raw, pers_to_pano_raw, pano_to_pers_accum_raw
from utils.image_utils import save_tensor, pil_to_torch
from utils.print_utils import print_warning

def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles: float = 0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)

class PanoramaModel(ImageModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config(ImageModel.Config):
        channels: int = 3
        pano_height: int = 2048
        pano_width: int = 4096
        initialization: str = "random"  # random, zero, gray, image
        init_img_path: Optional[str] = None

        learning_rate: float = 0.1
        eval_pos: Optional[int] = None
        max_steps: int = 10000

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None

    def prepare_optimization(self) -> None:
        self.image = torch.nn.Parameter(
            self.initialize_image(
                self.cfg.channels,
                self.cfg.pano_height,
                self.cfg.pano_width,
                self.cfg.initialization,
                self.cfg.init_img_path,
            )
        )
        self.preserve_mask = None
        print(self.cfg.init_img_path, self.cfg.initialization)
        if self.cfg.init_img_path is not None and self.cfg.initialization == "image":
            self.preserve_img = self.image.clone()
            self.preserve_mask = self.image.norm(dim=0, keepdim=True) > 1e-6
            self.preserve_mask = self.preserve_mask.expand_as(self.image)
            print(self.preserve_mask.sum())
        self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)
        # self.optimizer = torch.optim.AdamW([self.image], lr=self.cfg.learning_rate, weight_decay=0)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 100, int(self.cfg.max_steps*1.5))
        

    @torch.no_grad()
    def save(self, path: str) -> None:
        image = self.render_self()
        save_tensor(image, path)
    # @torch.no_grad()
    # def save(self, path: str) -> None:
    #     if self.cfg.channels == 3:
    #         img = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)
    #         # crop based on gray-color search
    #         print_warning('Manually cropping panorama image before saving. This is a temporary solution.')
    #         img_col = img[0, :, :, 0]
    #         first_pixel = img_col[:, 0:1]
    #         # search the first non-gray pixel index
    #         first_non_gray_idx = torch.where(img_col != first_pixel)[-1][0]
    #         last_non_gray_idx = torch.where(img_col != first_pixel)[-1][-1]
    #         img = img[:, :, first_non_gray_idx:last_non_gray_idx + 1, :]
    #         save_tensor(img, path)
    #     elif self.cfg.channels == 4:
    #         latent = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)
    #         img = shared_modules.prior.decode_latent(latent)
    #         save_tensor(img, path)
    #     else:
    #         raise ValueError(f"Channels must be 3 or 4, got {self.cfg.channels}")
    
    def render(self, camera) -> torch.Tensor:
        num_cameras = camera["num"]
        width, height, fov, azim, elev = (
            camera["width"],
            camera["height"],
            camera["fov"],
            camera["azimuth"],
            camera["elevation"],
        )

        img_projected = []
        for i in range(num_cameras):
            img_projected.append(
                pano_to_pers_raw(
                    self.image.unsqueeze(0),
                    fov,
                    azim[i],
                    elev[i],
                    height,
                    width,
                )
            )
        img_projected = torch.cat(img_projected, dim=0)

        return {
            "image": img_projected,
            "alpha": torch.ones(num_cameras, 1, height, width, device=self.cfg.device),
        }
        
    @torch.no_grad()
    def render_eval(self, path) -> torch.Tensor:
        image = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)

        elevs = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        azims = self.cfg.eval_pos
        if azims is None:
            azims = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324]
        
        dists = [1.5] * len(elevs)
        cameras = shared_modules.dataset.params_to_cameras(
            dists,
            elevs,
            azims,
        )
        images = self.render(cameras)["image"]
        latents = shared_modules.prior.encode_image_if_needed(images)
        rgbs = shared_modules.prior.decode_latent(latents)
        rgbs.clip_(0, 1)
        
        fns = [f"{azi}_{_i}" for _i, azi in enumerate(azims)]
        # Save perspective view images 09.10
        save_tensor(rgbs, path, fns=fns)
    
    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        image = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)

        # print_warning("Directly returning the raw image for stability. This is a temporary solution.")
        # return image
        
        elevs = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        azims = (0, 36, 72, 108, 144, 180, 216, 252, 288, 324)
        
        dists = [1.5] * len(elevs)
        cameras = shared_modules.dataset.params_to_cameras(
            dists,
            elevs,
            azims,
        )
        images = self.render(cameras)["image"]
        latents = shared_modules.prior.encode_image_if_needed(images)
        rgbs = shared_modules.prior.decode_latent(latents)

        # unprojection
        num_cameras = cameras["num"]
        width, height, fov, azim, elev = (
            cameras["width"],
            cameras["height"],
            cameras["fov"],
            cameras["azimuth"],
            cameras["elevation"],
        )

        img_new = torch.zeros(1, 3, 2048, 4096, device=self.cfg.device)
        img_cnt = torch.zeros(1, 1, 2048, 4096, device=self.cfg.device)
        for i in range(num_cameras):
            img_tmp, mask = pers_to_pano_raw(
                rgbs[i:i+1],
                fov,
                azim[i],
                elev[i],
                2048,
                4096,
                return_mask=True,
            )
            img_new += img_tmp.squeeze(0)
            img_cnt += mask.squeeze(0).long()
        
        image = img_new / (img_cnt + 1e-6)

        return image

    @torch.no_grad()
    def closed_form_optimize(self, step, camera, target):
        if self.image.shape[0] == 3:
            target = shared_modules.prior.decode_latent_if_needed(target)
        elif self.image.shape[0] == 4:
            target = shared_modules.prior.encode_image_if_needed(target)

        num_cameras = camera["num"]
        width, height, fov, azim, elev = (
            camera["width"],
            camera["height"],
            camera["fov"],
            camera["azimuth"],
            camera["elevation"],
        )

        img_new = torch.zeros_like(self.image)
        img_cnt = torch.zeros_like(self.image, dtype=torch.long)
        for i in range(num_cameras):
            img_tmp, mask = pers_to_pano_raw(
                target[i:i+1],
                fov,
                azim[i],
                elev[i],
                self.cfg.pano_height,
                self.cfg.pano_width,
                return_mask=True,
            )
            img_new += img_tmp.squeeze(0)
            # round mask
            img_cnt += mask.squeeze(0).long()
        
        img_new = img_new / (img_cnt + 1e-6)
        img_new[img_cnt == 0] = self.image[img_cnt == 0]

        if self.preserve_mask is not None:
            print_info("Preserving the original image...")
            img_new[self.preserve_mask] = self.preserve_img[self.preserve_mask]
        
        self.image.data = img_new
    

    def get_noise(self, camera):
        noise_map = torch.randn(1, 4, 2048, 4096, device=self.cfg.device)
        num_cameras = camera["num"]
        fov, azim, elev = (
            camera["fov"],
            camera["azimuth"],
            camera["elevation"],
        )
        _, _, height, width = shared_modules.prior.latent_res

        noise_projected = []
        for i in range(num_cameras):
            pers, cnt = pano_to_pers_accum_raw(noise_map, fov, azim[i], elev[i], height, width)
            pers = pers / (torch.sqrt(cnt) + 1e-8)
            noise_projected.append(pers)
        noise_projected = torch.cat(noise_projected, dim=0)

        return noise_projected
        