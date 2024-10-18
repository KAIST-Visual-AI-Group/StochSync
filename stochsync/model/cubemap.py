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
from utils.panorama_utils import  xy_to_lonlat, lonlat_to_xy, xy_to_lonlat_plane, lonlat_to_xy_plane
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

class CubemapModel(ImageModel):
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

        learning_rate: float = 0.1
        eval_pos: Optional[int] = None
        max_steps: int = 10000

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None
        self.scheduler = None 

    def prepare_optimization(self) -> None:
        self.mask = {}
        self.image = {}
        for direction in ["front", "back", "left", "right", "top", "bottom"]:
            self.image[direction] = torch.nn.Parameter(
                self.initialize_image(
                    self.cfg.channels,
                    self.cfg.height,
                    self.cfg.width,
                    self.cfg.initialization,
                    self.cfg.init_img_path,
                )
            )
            self.mask[direction] = torch.zeros(1, self.cfg.height, self.cfg.width, device=self.cfg.device, dtype=torch.bool)
        self.optimizer = torch.optim.AdamW([self.image[key] for key in self.image], lr=self.cfg.learning_rate)
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
        assert all([az in [0, 45, 90, 135, 180, 225, 270, 315] for az in azim])
        assert all([el in [-90, -45, 0, 45, 90] for el in elev])
        easydict = {
            0: "front",
            90: "right",
            180: "back",
            270: "left",
            360: "front",
        }


        mask_projected = []
        img_projected = []
        for i in range(num_cameras):
            el, az = elev[i], azim[i]
            if el == 0:
                if az in easydict:
                    img_tmp = self.image[easydict[az]]
                    mask_tmp = self.mask[easydict[az]]
                else:
                    img_tmp = torch.zeros_like(self.image["front"])
                    mask_tmp = torch.zeros_like(img_tmp)
                    assert az - 45 in easydict and az + 45 in easydict
                    img_tmp[..., :width//2] = self.image[easydict[az - 45]][..., width//2:]
                    img_tmp[..., width//2:] = self.image[easydict[az + 45]][..., :width//2]
                    mask_tmp[..., :width//2] = self.mask[easydict[az - 45]][..., width//2:]
                    mask_tmp[..., width//2:] = self.mask[easydict[az + 45]][..., :width//2]
            elif el == 90:
                img_tmp = torch.rot90(self.image["top"], - az // 90, [1, 2])
                mask_tmp = torch.rot90(self.mask["top"], - az // 90, [1, 2])
            elif el == -90:
                img_tmp = torch.rot90(self.image["bottom"], az // 90, [1, 2])
                mask_tmp = torch.rot90(self.mask["bottom"], az // 90, [1, 2])
            elif el == 45:
                img_tmp = torch.zeros_like(self.image["front"])
                img_tmp[..., :height//2, :] = torch.rot90(self.image["top"], - az // 90, [1, 2])[..., height//2:, :]
                assert az in easydict
                img_tmp[..., height//2:, :] = self.image[easydict[az]][..., :height//2, :]
                mask_tmp = torch.zeros_like(img_tmp)
                mask_tmp[..., :height//2, :] = torch.rot90(self.mask["top"], - az // 90, [1, 2])[..., height//2:, :]
                mask_tmp[..., height//2:, :] = self.mask[easydict[az]][..., :height//2, :]
            elif el == -45:
                img_tmp = torch.zeros_like(self.image["front"])
                img_tmp[..., height//2:, :] = torch.rot90(self.image["bottom"], az // 90, [1, 2])[..., :height//2, :]
                assert az in easydict
                img_tmp[..., :height//2, :] = self.image[easydict[az]][..., height//2:, :]
                mask_tmp = torch.zeros_like(img_tmp)
                mask_tmp[..., height//2:, :] = torch.rot90(self.mask["bottom"], az // 90, [1, 2])[..., :height//2, :]
                mask_tmp[..., :height//2, :] = self.mask[easydict[az]][..., height//2:, :]

            img_projected.append(img_tmp)
            mask_projected.append(mask_tmp)

        img_projected = torch.stack(img_projected, dim=0)
        mask_projected = torch.stack(mask_projected, dim=0)
        return {
            "image": img_projected,
            "alpha": mask_projected.float()
        }
        
    @torch.no_grad()
    def render_eval(self, path) -> torch.Tensor:
        return
    
    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        H, W = self.cfg.height, self.cfg.width
        image = torch.zeros(1, 3, H * 3, W * 4, device=self.cfg.device)
        image[:, :, H:2*H, 0:W] = self.image["left"]
        image[:, :, H:2*H, W:2*W] = self.image["front"]
        image[:, :, H:2*H, 2*W:3*W] = self.image["right"]
        image[:, :, H:2*H, 3*W:4*W] = self.image["back"]
        image[:, :, 0:H, W:2*W] = self.image["top"]
        image[:, :, 2*H:3*H, W:2*W] = self.image["bottom"]

        return image
    

    def get_diffusion_softmask(self, camera):
        # num_cameras = camera["num"]
        # width, height = (
        #     camera["width"],
        #     camera["height"]
        # )
        
        # H, W = height, width
        # x = torch.linspace(0, 1, W // 2)
        # x = torch.cat([x, torch.flip(x, dims=[0])])
        # y = torch.linspace(0, 1, H // 2)
        # y = torch.cat([y, torch.flip(y, dims=[0])])
        # mask_x = x.unsqueeze(0).repeat(H, 1)
        # mask_y = y.unsqueeze(1).repeat(1, W)
        # mask = torch.max(mask_x, mask_y).to(self.cfg.device)
        # mask = mask.unsqueeze(0).unsqueeze(0).repeat(num_cameras, 1, 1, 1)
        # return mask
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
        return mask

    @torch.no_grad()
    def closed_form_optimize(self, step, camera, target):
        if self.image['front'].shape[0] == 3:
            target = shared_modules.prior.decode_latent_if_needed(target)
        elif self.image['front'].shape[0] == 4:
            target = shared_modules.prior.encode_image_if_needed(target)

        num_cameras = camera["num"]
        width, height, fov, azim, elev = (
            camera["width"],
            camera["height"],
            camera["fov"],
            camera["azimuth"],
            camera["elevation"],
        )

        assert all([az in [0, 45, 90, 135, 180, 225, 270, 315] for az in azim])
        assert all([el in [-90, -45, 0, 45, 90] for el in elev])
        easydict = {
            0: "front",
            90: "right",
            180: "back",
            270: "left",
            360: "front",
        }
        easyTS = {
            0: (slice(height//2, None), slice(None)),
            90: (slice(None), slice(width//2, None)),
            180: (slice(None, height//2), slice(None)),
            270: (slice(None), slice(None, width//2)),
        }
        easyBS = {
            0: (slice(None, height//2), slice(None)),
            90: (slice(None), slice(width//2, None)),
            180: (slice(height//2, None), slice(None)),
            270: (slice(None), slice(None, width//2)),
        }

        img_new = {key: torch.zeros_like(self.image[key]) for key in self.image}
        img_cnt = {key: torch.zeros_like(self.image[key], dtype=torch.long) for key in self.image}

        for i, tgt in enumerate(target):
            el, az = elev[i], azim[i]
            if el == 0:
                if az in easydict:
                    img_new[easydict[az]] += tgt
                    img_cnt[easydict[az]] += 1
                else:
                    assert az - 45 in easydict and az + 45 in easydict
                    img_new[easydict[az - 45]][..., width//2:] += tgt[..., :width//2]
                    img_new[easydict[az + 45]][..., :width//2] += tgt[..., width//2:]
                    img_cnt[easydict[az - 45]][..., width//2:] += 1
                    img_cnt[easydict[az + 45]][..., :width//2] += 1
            elif el == 90:
                img_new["top"] += torch.rot90(tgt, az // 90, [1, 2])
                img_cnt["top"] += 1
            elif el == -90:
                img_new["bottom"] += torch.rot90(tgt, -az // 90, [1, 2])
                img_cnt["bottom"] += 1
            elif el == 45:
                img_new["top"][..., easyTS[az][0], easyTS[az][1]] += torch.rot90(tgt[..., :height//2, :], az // 90, [1, 2])
                img_new[easydict[az]][..., :height//2, :] += tgt[..., height//2:, :]
                img_cnt["top"][..., easyTS[az][0], easyTS[az][1]] += 1
                img_cnt[easydict[az]][..., :height//2, :] += 1
            elif el == -45:
                img_new["bottom"][..., easyBS[az][0], easyTS[az][1]] += torch.rot90(tgt[..., height//2:, :], -az // 90, [1, 2])
                img_new[easydict[az]][..., height//2:, :] += tgt[..., :height//2, :]
                img_cnt["bottom"][..., easyBS[az][0], easyTS[az][1]] += 1
                img_cnt[easydict[az]][..., height//2:, :] += 1

        for key in img_new:
            img_new[key] = img_new[key] / (img_cnt[key].float() + 1e-6)
            img_new[key][img_cnt[key] == 0] = self.image[key][img_cnt[key] == 0]
            self.image[key].data = img_new[key]
            self.mask[key] |= img_cnt[key][:1] > 0
    

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
        