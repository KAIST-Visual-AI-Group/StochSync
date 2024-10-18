from typing import Dict, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
import diffusers
from diffusers import StableDiffusionInpaintPipeline
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionGLIGENPipeline,
    DDIMScheduler,
)
from diffusers import (
    StableDiffusionXLPipeline,
    MarigoldDepthPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    DDIMScheduler,
)
from third_party.mvdream.pipeline_mvdream import MVDreamPipeline

from ..utils.camera_utils import convert_camera_convention
from ..utils.extra_utils import (
    attach_direction_prompt,
    ignore_kwargs,
    attach_elevation_prompt,
)
from ..utils.print_utils import print_info, print_warning, print_error
from .. import shared_modules as sm
from .base import Prior, NEGATIVE_PROMPT
from .sd import StableDiffusionPrior


class InpaintingPrior(StableDiffusionPrior):
    def __init__(self, cfg):
        self.cfg = self.Config(**cfg)

        if not ((self.cfg.width == self.cfg.height == 512) or (self.cfg.width == self.cfg.height == 64)):
            print_error("Width and height must be 512(64) for Stable Diffusion")
            raise ValueError

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler"
        )
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            scheduler=self.scheduler,
            torch_dtype=torch.float16 if self.cfg.mixed_precision else torch.float32,
        ).to("cuda")
        self.scheduler.set_timesteps(30)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        # self.cfg.text_prompt = self.cfg.text_prompt + ", best quality, high quality, extremely detailed, good geometry"
        print_info(self.cfg.text_prompt)
    
    @property
    def rgb_res(self):
        return 1, 3, 512, 512
    
    @property
    def latent_res(self):
        return 1, 4, 64, 64

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None, mask_image_latents=None, mask=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        text_prompts = [text_prompt] * camera["num"]

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

        if mask_image_latents is not None and mask is not None:
            self.cond_image = mask_image_latents
            self.cond_mask = mask
            return self.cond
        render_pkg = sm.model.render(camera)
        images = render_pkg["image"]
        mask = 1 - render_pkg["alpha"][:, :1]

        init_image = sm.prior.pipeline.image_processor.preprocess(
            images, height=images.shape[-2], width=images.shape[-1]
        )
        mask_condition = sm.prior.pipeline.mask_processor.preprocess(mask, height=mask.shape[-2], width=mask.shape[-1])
        masked_image = init_image * (mask_condition < 0.5)

        mask, masked_image_latents = sm.prior.pipeline.prepare_mask_latents(
            mask_condition,
            masked_image,
            1,
            512,
            512,
            torch.float32,
            "cuda",
            generator=None,
            do_classifier_free_guidance=True
        )

        self.cond_image = masked_image_latents[:1]
        self.cond_mask = mask[:1]

        # self.cond_image = self.encode_image(images * (mask < 0.5))  # B 4 64 64
        # self.cond_mask = F.interpolate(mask, size=(64, 64), mode="nearest")  # B 1 64 64

        return self.cond

    def sample(self, camera, text_prompt=None):
        if text_prompt is None:
            text_prompt = self.cfg.text_prompt

        self.prepare_cond(camera)
        with torch.no_grad():
            images = self.pipeline(
                [text_prompt], negative_prompt=[self.cfg.negative_prompt]
            ).images
        return images

    def predict(self, camera, x_t, timestep, guidance_scale=None, return_dict=False, text_prompt=None, negative_prompt=None, mask_image_latents=None, mask=None):
        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        self.prepare_cond(camera, text_prompt, negative_prompt, mask_image_latents, mask)
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        x_t = torch.cat([x_t, self.cond_mask, self.cond_image], dim=1)
        x_t_stack = torch.cat([x_t] * 2)
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