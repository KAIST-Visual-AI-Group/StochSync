from typing import Dict, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import diffusers
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionGLIGENPipeline,
    DDIMScheduler,
)
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers import (
    StableDiffusionXLPipeline,
    MarigoldDepthPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    DDIMScheduler,
)
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    DiffusionPipeline,
    StableDiffusionPipeline,
)
# from pipeline.controlnet_wrapper import SyncTweediesControlNet

from third_party.mvdream.pipeline_mvdream import MVDreamPipeline

from .base import Prior, NEGATIVE_PROMPT
from utils.camera_utils import convert_camera_convention, camera_hash
from utils.extra_utils import (
    attach_direction_prompt,
    attach_detailed_direction_prompt,
    ignore_kwargs,
    attach_elevation_prompt,
)
from k_utils.print_utils import print_info, print_warning, print_error

import shared_modules as sm


def preprocess_depth(depth, mask):
    disp = depth.clone()
    disp[mask] = 1 / (disp[mask] + 1e-15)
    _min = disp[mask].min()
    _max = disp[mask].max()
    disp[mask] = (disp[mask] - _min) / (_max - _min)
    disp[~mask] = 0
    return disp


class SD2DepthPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        model_name: str = "stabilityai/stable-diffusion-2-depth"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = NEGATIVE_PROMPT
        width: int = 512
        height: int = 512
        guidance_scale: int = 100
        mixed_precision: bool = False
        root_dir: str = "./results/default"
        use_view_dependent_prompt: bool = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        if not (
            (self.cfg.width == self.cfg.height == 768)
            or (self.cfg.width == self.cfg.height == 96)
        ):
            print_error("Width and height must be 768(96) for Stable Diffusion")
            raise ValueError

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.model_name, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
            self.cfg.model_name,
            scheduler=self.scheduler,
            torch_dtype=torch.float16 if self.cfg.mixed_precision else torch.float32,
        ).to("cuda")
        self.scheduler.set_timesteps(30)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        self.depth = None
        self.prev_camera_hash = -65535

    @property
    def rgb_res(self):
        return 1, 3, 768, 768

    @property
    def latent_res(self):
        return 1, 4, 96, 96

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = (
            negative_prompt if negative_prompt is not None else self.cfg.negative_prompt
        )

        if self.cfg.use_view_dependent_prompt:
            text_prompts = attach_detailed_direction_prompt(
                self.cfg.text_prompt, camera["elevation"], camera["azimuth"]
            )
        else:
            text_prompts = [self.cfg.text_prompt] * camera["num"]

        neg_embeds, pos_embeds = [], []
        for prompt in text_prompts:
            text_embeddings = self.encode_text(
                prompt, negative_prompt=negative_prompt
            )  # neg, pos
            neg, pos = text_embeddings.chunk(2)
            neg_embeds.append(neg)
            pos_embeds.append(pos)

        text_embeddings = torch.cat(neg_embeds + pos_embeds)

        self.cond = {"encoder_hidden_states": text_embeddings}

        # Depth caching for efficient ODE solving
        cam_hash = camera_hash(camera)
        if cam_hash != self.prev_camera_hash:
            rendered_pkg = sm.model.render(camera, bsdf="depth")
            depth = rendered_pkg["image"][:, 0:1]
            mask = rendered_pkg["alpha"] > 0.99
            disp = preprocess_depth(depth, mask)
            W = self.latent_res[2]
            disp = torch.nn.functional.interpolate(disp, (W, W), mode="bicubic")
            disp = 2 * (disp - disp.min()) / (disp.max() - disp.min()) - 1
            self.depth = torch.cat([disp] * 2).to(self.dtype)
            self.prev_camera_hash = cam_hash

        return self.cond

    def sample(self, text_prompt, num_samples=1):
        raise NotImplementedError

    def predict(
        self,
        camera,
        x_t,
        timestep,
        guidance_scale=None,
        return_dict=False,
        text_prompt=None,
        negative_prompt=None,
    ):
        # Predict the noise using the UNet model
        x_t = self.encode_image_if_needed(x_t)
        # cast if needed
        x_t = x_t.to(self.dtype)

        self.prepare_cond(camera, text_prompt, negative_prompt)
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        x_t_stack = torch.cat([x_t] * 2)
        x_t_stack = torch.cat([x_t_stack, self.depth], dim=1)
        noise_pred = self.pipeline.unet(x_t_stack, timestep, **self.cond).sample
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


class ControlNetPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        model_name: str = "stabilityai/stable-diffusion-2-depth"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = 'oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.'
        width: int = 512
        height: int = 512
        guidance_scale: int = 100
        mixed_precision: bool = False
        root_dir: str = "./results/default"
        use_view_dependent_prompt: bool = False

        conditioning_scale: float = 0.8

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        if not (
            (self.cfg.width == self.cfg.height == 768)
            or (self.cfg.width == self.cfg.height == 96)
        ):
            print_error("Width and height must be 768(96) for Stable Diffusion")
            raise ValueError

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.model_name, subfolder="scheduler"
        )

        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float16 if self.cfg.mixed_precision else torch.float32,
        ).to("cuda")
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            torch_dtype=torch.float16 if self.cfg.mixed_precision else torch.float32,
        ).to("cuda")

        self.scheduler.set_timesteps(30)
        self.controlnet.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)

        self.depth = None
        self.prev_camera_hash = -65535

    @property
    def rgb_res(self):
        return 1, 3, 768, 768

    @property
    def latent_res(self):
        return 1, 4, 96, 96

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = (
            negative_prompt if negative_prompt is not None else self.cfg.negative_prompt
        )

        if self.cfg.use_view_dependent_prompt:
            text_prompts = attach_detailed_direction_prompt(
                self.cfg.text_prompt, camera["elevation"], camera["azimuth"]
            )
            # print(text_prompts)
        else:
            text_prompts = [self.cfg.text_prompt] * camera["num"]

        neg_embeds, pos_embeds = [], []
        for prompt in text_prompts:
            text_embeddings = self.encode_text(
                prompt, negative_prompt=negative_prompt
            )  # neg, pos
            neg, pos = text_embeddings.chunk(2)
            neg_embeds.append(neg)
            pos_embeds.append(pos)

        text_embeddings = torch.cat(neg_embeds + pos_embeds)

        self.cond = {"encoder_hidden_states": text_embeddings}


        cam_hash = camera_hash(camera)
        if cam_hash != self.prev_camera_hash:
            rendered_pkg = sm.model.render(camera, bsdf="depth")
            depth = rendered_pkg["image"][:, 0:1]
            mask = rendered_pkg["alpha"] > 0.99
            disp = preprocess_depth(depth, mask)
            # disp = torch.nn.functional.interpolate(disp, (64, 64), mode="bicubic")
            disp = (disp - disp.min()) / (disp.max() - disp.min())
            disp = torch.cat([disp.expand(-1, 3, -1, -1)] * 2)
            self.depth = disp.to(self.dtype)
            self.prev_camera_hash = cam_hash

        return self.cond

    def sample(self, text_prompt, num_samples=1):
        raise NotImplementedError

    def predict(
        self,
        camera,
        x_t,
        timestep,
        guidance_scale=None,
        return_dict=False,
        text_prompt=None,
        negative_prompt=None,
    ):
        # Predict the noise using the UNet model
        x_t = self.encode_image_if_needed(x_t)
        x_t = x_t.to(self.dtype)

        self.prepare_cond(camera, text_prompt, negative_prompt)
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        x_t_stack = torch.cat([x_t] * 2)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x_t_stack,
            timestep,
            controlnet_cond=self.depth,
            conditioning_scale=self.cfg.conditioning_scale,
            guess_mode=False,
            return_dict=False,
            **self.cond
        )
        noise_pred = self.pipeline.unet(
            x_t_stack,
            timestep,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            **self.cond
        ).sample

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
