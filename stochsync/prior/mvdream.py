from typing import Dict, Literal
from dataclasses import dataclass, field

import numpy as np
import torch
import diffusers
from diffusers import DDIMScheduler
from third_party.mvdream.pipeline_mvdream import MVDreamPipeline

from ..utils.camera_utils import convert_camera_convention
from ..utils.extra_utils import (
    attach_direction_prompt,
    ignore_kwargs,
)
from ..utils.print_utils import print_info, print_warning, print_error

from .base import Prior, NEGATIVE_PROMPT


# Set the logging verbosity to error to avoid unnecessary warnings
diffusers.logging.set_verbosity_error()

class MVDreamPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        model_name: str = "ashawkey/mvdream-sd2.1-diffusers"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = NEGATIVE_PROMPT
        width: int = 512
        height: int = 512
        guidance_scale: int = 100

        elevation: int = 0
        mixed_precision: bool = False
        use_view_dependent_prompt: bool = False

        convention: Literal[
            "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
        ] = "RDF"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        if not (
            (self.cfg.width == self.cfg.height == 256)
            or (self.cfg.width == self.cfg.height == 32)
        ):
            print_error("Width and height must be 256(32) for MVDream")
            raise ValueError

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.model_name, subfolder="scheduler"
        )
        self.pipeline = MVDreamPipeline.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch.float16 if self.cfg.mixed_precision else torch.float32,
            scheduler=self.scheduler,
            trust_remote_code=True,
        ).to("cuda")
        self.scheduler.set_timesteps(30)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

    @property
    def rgb_res(self):
        return 1, 3, 256, 256

    @property
    def latent_res(self):
        return 1, 4, 32, 32

    def prepare_cond(self, camera, mv_camera):
        assert len(camera["elevation"]) == len(mv_camera), (
            len(camera["elevation"]),
            len(mv_camera),
        )

        if self.cfg.use_view_dependent_prompt:
            text_prompts = attach_direction_prompt(
                self.cfg.text_prompt, camera["elevation"], camera["azimuth"]
            )
        else:
            text_prompts = [self.cfg.text_prompt] * camera["num"]

        neg_embeds, pos_embeds = [], []
        for prompt in text_prompts:
            text_embeddings = self.encode_text(
                prompt, negative_prompt=self.cfg.negative_prompt
            )  # neg, pos
            neg, pos = text_embeddings.chunk(2)
            neg_embeds.append(neg)
            pos_embeds.append(pos)

        text_embeddings = torch.stack(neg_embeds + pos_embeds)

        self.cond = {
            "context": text_embeddings,
            "num_frames": len(mv_camera),
            "camera": torch.cat([mv_camera] * 2),
        }

        return self.cond

    def sample(self, text_prompt=None, camera=None):
        if text_prompt is None:
            text_prompt = self.cfg.text_prompt

        with torch.no_grad():
            images = self.pipeline(
                [text_prompt],
                negative_prompt=[self.cfg.negative_prompt],
                elevation=self.cfg.elevation,
                camera=camera,
            )
        return images

    def predict(
        self, camera, x_t, timestep, guidance_scale=None, return_dict=False, cond=None
    ):
        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        batch_size = x_t.shape[0]
        assert batch_size % 4 == 0, "Batch size must be a multiple of 4 for MVDream"

        # Get camera
        mv_camera = convert_camera_convention(
            camera["c2w"].cpu().numpy(), self.cfg.convention, "OpenGL"
        )  # B 4 4
        mv_camera = torch.from_numpy(mv_camera).to(x_t.device)
        mv_camera[:, 0:3, 3] /= mv_camera[:, 0:3, 3].norm(dim=-1, keepdim=True) + 1e-8
        mv_camera = mv_camera.view(-1, 16)  # B 16

        if cond is None:
            self.prepare_cond(camera, mv_camera)
        else:
            self.cond = cond

        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        x_t_stack = torch.cat([x_t] * 2)
        timesteps = torch.tensor([timestep] * batch_size * 2).to(x_t.device)

        noise_pred = self.pipeline.unet(x_t_stack, timesteps, **self.cond)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if return_dict:
            return {
                "noise_pred": noise_pred,
                "noise_pred_uncond": noise_pred_uncond,
                "noise_pred_text": noise_pred_text,
            }
        return noise_pred
