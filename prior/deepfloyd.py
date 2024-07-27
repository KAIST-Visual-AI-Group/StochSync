from functools import lru_cache
from typing import Dict

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionGLIGENPipeline,
    DDIMScheduler,
)
from diffusers import DiffusionPipeline
from diffusers import (
    StableDiffusionXLPipeline,
    MarigoldDepthPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    DDIMScheduler,
)

from .base import Prior, NEGATIVE_PROMPT
from utils.extra_utils import (
    attach_direction_prompt,
    ignore_kwargs,
)  # How to remove this project-specific import?

from k_utils.print_utils import print_info


class DeepFloydPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int
        height: int
        model_name: str = "DeepFloyd/IF-I-M-v1.0"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = NEGATIVE_PROMPT
        guidance_scale: int = 100
        mixed_precision: bool = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch.float16 if self.cfg.mixed_precision else torch.float32,
        ).to("cuda")

        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.unet.requires_grad_(False)

    def encode_image(self, img_tensor):
        return img_tensor

    def decode_latent(self, latent):
        return latent

    def prepare_cond(self, camera):
        text_prompts = [self.cfg.text_prompt] * camera["num"]

        neg_embeds, pos_embeds = [], []
        for prompt in text_prompts:
            text_embeddings = self.encode_text(
                prompt, negative_prompt=self.cfg.negative_prompt
            )  # neg, pos
            neg, pos = text_embeddings.chunk(2)
            neg_embeds.append(neg)
            pos_embeds.append(pos)

        text_embeddings = torch.cat(neg_embeds + pos_embeds)

        self.cond = {"encoder_hidden_states": text_embeddings}
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

    def predict(self, camera, x_t, timestep):
        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)
        x_t_stack = torch.cat([x_t] * 2)

        self.prepare_cond(camera)

        noise_preds = self.pipeline.unet(x_t_stack, timestep, **self.cond).sample
        noise_pred_uncond, noise_pred_text = noise_preds.chunk(2)
        noise_preds = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        return noise_preds