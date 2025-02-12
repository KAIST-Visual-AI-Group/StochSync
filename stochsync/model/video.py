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

class VideoModel(ImageModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config(ImageModel.Config):
        channels: int = 3
        height: int = 64
        width: int = 64
        initialization: str = "random"  # random, zero, gray, image
        init_img_path: Optional[str] = None

        learning_rate: float = 0.1
        eval_pos: Optional[int] = None
        max_steps: int = 10000
        seam_removal_mode: str = "horizontal"  # horizontal, vertical, both

        n_frames: int = 40
        root_dir: str = "./results/default"

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None


    def prepare_optimization(self) -> None:
        self.image = torch.nn.Parameter(
            torch.randn(self.cfg.n_frames, self.cfg.channels, self.cfg.height, self.cfg.width, device=self.cfg.device)
        )
        
        self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)
        

    @torch.no_grad()
    def save(self, path: str) -> None:
        image = self.render_self()
        save_tensor(image, path, save_type="video", fps=8)
    
    def render(self, camera) -> torch.Tensor:
        num_cameras = camera["num"]
        width, height, start_frames, end_frames = (
            camera["width"],
            camera["height"],
            camera["start_frames"],
            camera["end_frames"],
        )

        video_batch = []
        for i in range(num_cameras):
            start_frame = start_frames[i]
            end_frame = end_frames[i]
            video_clip = self.image[start_frame:end_frame]
            video_batch.append(video_clip)
        video_batch = torch.stack(video_batch, dim=0)
        mask_batch = torch.ones_like(video_batch)

        return {
            "image": video_batch,
            "alpha": mask_batch,
        }
        
    @torch.no_grad()
    def render_eval(self, path) -> torch.Tensor:
        pass
    
    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        return self.image
    

    def get_diffusion_softmask(self, camera):
        num_cameras = camera["num"]
        width, height = (
            camera["width"],
            camera["height"]
        )
        
        H, W = height, width
        F = camera["end_frames"][0] - camera["start_frames"][0]
        x = torch.linspace(0, 1, F//2)  # from 0 to 1 in 32 steps
        x = torch.cat([x, torch.flip(x, dims=[0])])  # mirror it to go back to 0 (64 steps in total)

        # mask to shape (F, C, H, W,) where C, H, W are the extended dimensions
        mask = x.view(F, 1, 1, 1).expand(F, 1, H, W)

        return mask

    @torch.no_grad()
    def closed_form_optimize(self, step, camera, target):
        if self.image.shape[0] == 3:
            target = shared_modules.prior.decode_latent_if_needed(target)
        elif self.image.shape[0] == 4:
            target = shared_modules.prior.encode_image_if_needed(target)
        

        num_cameras = camera["num"]
        width, height, start_frames, end_frames = (
            camera["width"],
            camera["height"],
            camera["start_frames"],
            camera["end_frames"],
        )
        
        video_batch = torch.zeros_like(self.image)
        cnt_batch = torch.zeros_like(self.image)
        for i in range(num_cameras):
            start_frame = start_frames[i]
            end_frame = end_frames[i]
            video_batch[start_frame:end_frame] += target
            cnt_batch[start_frame:end_frame] += 1

        video_batch[cnt_batch == 0] = self.image[cnt_batch == 0]
        cnt_batch[cnt_batch == 0] = 1
        video_batch = video_batch / (cnt_batch + 1e-8)
        
        self.image.data = video_new
    

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
        