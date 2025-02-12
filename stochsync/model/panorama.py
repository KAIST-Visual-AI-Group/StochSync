import os
from dataclasses import dataclass
from typing import Dict, Optional, List
import math

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image

from .. import shared_modules
from ..utils.extra_utils import ignore_kwargs
from ..utils.panorama_utils import pano_to_pers_raw, pers_to_pano_raw, pano_to_pers_accum_raw
from ..utils.panorama_utils import  compute_pano2pers_map, compute_pers2pano_map, compute_sp2pers_map, compute_pers2sp_map, compute_pers2torus_map, compute_torus2pers_map
from ..utils.image_utils import save_tensor, pil_to_torch
from ..utils.print_utils import print_warning, print_info

from .image import ImageModel

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
        mode: str = "panorama"  # panorama, sphere, torus
        seam_removal_mode: str = "horizontal"  # horizontal, vertical, both

        gt_image: Optional[str] = None
        gt_elev: Optional[float] = None
        gt_azim: Optional[float] = None

        root_dir: str = "./results/default"

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None
        self.scheduler = None 

        if self.cfg.mode == "panorama":
            self.c2i_func = compute_pano2pers_map
            self.i2c_func = compute_pers2pano_map
        elif self.cfg.mode == "sphere":
            self.c2i_func = compute_sp2pers_map
            self.i2c_func = compute_pers2sp_map
        elif self.cfg.mode == "torus":
            self.c2i_func = compute_torus2pers_map
            self.i2c_func = compute_pers2torus_map


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
        self.original_img = self.image.clone()

        self.gt_image = None
        if self.cfg.gt_image is not None:
            self.gt_image = Image.open(self.cfg.gt_image).convert("RGB")
            self.gt_image = pil_to_torch(self.gt_image).to(self.image)
        
        self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 100, int(self.cfg.max_steps*1.5))
        

    @torch.no_grad()
    def save(self, path: str) -> None:
        image = self.render_self()
        save_tensor(image, path)
    
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
        mask_projected = []
        mask = (self.image != self.original_img).float()
        for i in range(num_cameras):
            img_projected.append(
                pano_to_pers_raw(
                    self.image.unsqueeze(0),
                    fov,
                    azim[i],
                    elev[i],
                    height,
                    width,
                    mapping_func=self.c2i_func,
                    quat=camera.get("quat", None),
                )
            )
            mask_projected.append(
                pano_to_pers_raw(
                    mask.unsqueeze(0),
                    fov,
                    azim[i],
                    elev[i],
                    height,
                    width,
                    mapping_func=self.c2i_func,
                    quat=camera.get("quat", None),
                )
            )
        img_projected = torch.cat(img_projected, dim=0)
        mask_projected = torch.cat(mask_projected, dim=0)

        return {
            "image": img_projected,
            "alpha": mask_projected,
        }
        
    @torch.no_grad()
    def render_eval(self, path) -> torch.Tensor:
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
        rgbs = shared_modules.prior.decode_latent_if_needed(images)
        rgbs.clip_(0, 1)
        
        fns = [f"{azi}_{_i}" for _i, azi in enumerate(azims)]
        # Save perspective view images 09.10
        save_tensor(rgbs, path, fns=fns)
    
    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        image = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)

        # print_info("Directly returning the raw image for stability.")
        return image
        
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

        img_new = torch.zeros(1, 3, self.cfg.pano_height, self.cfg.pano_width, device=self.cfg.device)
        img_cnt = torch.zeros(1, 1, self.cfg.pano_height, self.cfg.pano_width, device=self.cfg.device)
        for i in range(num_cameras):
            img_tmp, mask = pers_to_pano_raw(
                rgbs[i:i+1],
                fov,
                azim[i],
                elev[i],
                self.cfg.pano_height,
                self.cfg.pano_width,
                return_mask=True,
                xy_to_lonlat=self.xy_to_lonlat,
            )
            img_new += img_tmp.squeeze(0)
            img_cnt += mask.squeeze(0).long()
        
        image = img_new / (img_cnt + 1e-6)

        return image
    

    def get_diffusion_softmask(self, camera):
        num_cameras = camera["num"]
        width, height = (
            camera["width"],
            camera["height"]
        )
        
        H, W = height, width
        x = torch.linspace(0, 1, W//2)  # from 0 to 1 in 32 steps
        x = torch.cat([x, torch.flip(x, dims=[0])])  # mirror it to go back to 0 (64 steps in total)
        mask = x.repeat(H, 1).to(self.device)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(num_cameras, 1, 1, 1)

        if self.cfg.seam_removal_mode == "horizontal":
            return mask
        elif self.cfg.seam_removal_mode == "vertical":
            return mask.permute(0, 1, 3, 2)
        elif self.cfg.seam_removal_mode == "both":
            h_mask = mask
            v_mask = mask.permute(0, 1, 3, 2)
            # take minimum
            return torch.min(h_mask, v_mask)

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
            if self.gt_image is not None and azim[i] == self.cfg.gt_azim and elev[i] == self.cfg.gt_elev:
                print_info("GT image found!")
                target[i] = F.interpolate(self.gt_image, size=(height, width), mode="bilinear", align_corners=False)[i]
            img_tmp, mask = pers_to_pano_raw(
                target[i:i+1],
                fov,
                azim[i],
                elev[i],
                self.cfg.pano_height,
                self.cfg.pano_width,
                return_mask=True,
                mapping_func=self.i2c_func,
                quat=camera.get("quat", None),
            )
            img_new += img_tmp.squeeze(0)
            # round mask
            img_cnt += mask.unsqueeze(0).expand(3,-1,-1).long()
        
        img_new = img_new / (img_cnt + 1e-6)
        img_new[img_cnt == 0] = self.image[img_cnt == 0]
        
        self.image.data = img_new
    

    def get_noise(self, camera):
        noise_map = torch.randn(1, 4, self.cfg.pano_height, self.cfg.pano_width, device=self.cfg.device)
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
        