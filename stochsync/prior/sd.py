from typing import Dict, Literal, Tuple, List
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

from .base import Prior, NEGATIVE_PROMPT


class StableDiffusionPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        model_name: str = "stabilityai/stable-diffusion-2-1-base"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = NEGATIVE_PROMPT
        width: int = 512
        height: int = 512
        guidance_scale: int = 100
        mixed_precision: bool = False
        root_dir: str = "./results/default"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        # if not ((self.cfg.width == self.cfg.height == 512) or (self.cfg.width == self.cfg.height == 64)):
        #     print_error("Width and height must be 512(64) for Stable Diffusion")
        #     raise ValueError

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.model_name, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.model_name,
            scheduler=self.scheduler,
            torch_dtype=torch.float16 if self.cfg.mixed_precision else torch.float32,
        ).to("cuda")
        self.scheduler.set_timesteps(30)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        print_info(self.cfg.text_prompt)
        
    @property
    def rgb_res(self):
        return 1, 3, 512, 512
    
    @property
    def latent_res(self):
        return 1, 4, 64, 64

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
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

    def predict(self, camera, x_t, timestep, guidance_scale=None, return_dict=False, text_prompt=None, negative_prompt=None):
        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        self.prepare_cond(camera, text_prompt, negative_prompt)
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

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

class UltimateStableDiffusionPrior(StableDiffusionPrior):
    @ignore_kwargs
    @dataclass
    class Config(StableDiffusionPrior.Config):
        angle_prompts: List[Tuple[int, int, str]] = field(
            default_factory=lambda: {
                # (azim, elev, prompt)
                (0, 0): "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
                (0, 90): "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
                (0, 180): "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
                (0, 270): "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
            }
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        def sphere_dist(azim, elev, azim2, elev2):
            return np.arccos(
                np.sin(np.radians(elev)) * np.sin(np.radians(elev2))
                + np.cos(np.radians(elev)) * np.cos(np.radians(elev2)) * np.cos(np.radians(azim - azim2))
            )

        azim, elev = camera["azimuth"], camera["elevation"]
        text_prompts = []
        for az, el in zip(azim, elev):
            # find the closest angle on the sphere
            min_dist = 1e9
            min_prompt = None
            for az2, el2, prompt in self.cfg.angle_prompts:
                dist = sphere_dist(az, el, az2, el2)
                if dist < min_dist:
                    min_dist = dist
                    min_prompt = prompt
            text_prompts.append(min_prompt)
        print(text_prompts)

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
        return self.cond

class CubemapStableDiffusionPrior(StableDiffusionPrior):
    @ignore_kwargs
    @dataclass
    class Config(StableDiffusionPrior.Config):
        angle_prompts: Dict[int, str] = field(
            default_factory=lambda: {
                "front": "a room",
                "right": "a room",
                "back": "a room",
                "left": "a room",
                "top": "a ceiling",
                "bottom": "a floor",
            }
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        azim, elev = camera["azimuth"], camera["elevation"]
        text_prompts = []
        for az, el in zip(azim, elev):
            if el == 90:
                text_prompts.append(self.cfg.angle_prompts["top"])
            elif el == -90:
                text_prompts.append(self.cfg.angle_prompts["bottom"])
            else:
                if az == 0 or az == 45:
                    text_prompts.append(self.cfg.angle_prompts["front"])
                elif az == 90 or az == 135:
                    text_prompts.append(self.cfg.angle_prompts["right"])
                elif az == 180 or az == 225:
                    text_prompts.append(self.cfg.angle_prompts["back"])
                elif az == 270 or az == 315:
                    text_prompts.append(self.cfg.angle_prompts["left"])
        # print(text_prompts)

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
        return self.cond

class ViewDependentStableDiffusionPrior(StableDiffusionPrior):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        text_prompts = attach_direction_prompt(
            text_prompt, camera["elevation"], camera["azimuth"]
        )

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
        return self.cond

class ElevDependentStableDiffusionPrior(StableDiffusionPrior):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = negative_prompt if negative_prompt is not None else self.cfg.negative_prompt

        text_prompts = attach_elevation_prompt(
            text_prompt, camera["elevation"]
        )

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
        return self.cond


class AngleDependentStableDiffusionPrior(StableDiffusionPrior):
    @ignore_kwargs
    @dataclass
    class Config(StableDiffusionPrior.Config):
        angle_prompts: Dict[int, str] = field(
            default_factory=lambda: {
                0: "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
                90: "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
                180: "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
                270: "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes",
            }
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def prepare_cond(self, camera):
        text_prompts = [self.cfg.angle_prompts[angle] for angle in camera["num"]]

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


class MarigoldPrior(Prior):
    def __init__(
        self, model_name="prs-eth/marigold-depth-lcm-v1-0", condition_parameters=None
    ):
        super().__init__()
        self.pipeline = MarigoldDepthPipeline.from_pretrained(model_name).to("cuda")

        self.condition_parameters = condition_parameters

    def sample(self, image, num_samples=1):
        with torch.no_grad():
            res = self.pipeline(image)
        return res

    def predict(self, x_t, timestep, text_prompt, condition_images):
        pass


class ControlNetPrior(Prior):
    def __init__(self, model_name="controlnet/controlnet", condition_parameters=None):
        super().__init__()
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_name
        ).to("cuda")

        self.condition_parameters = condition_parameters

    def sample(self, text_prompt, num_samples=1):
        with torch.no_grad():
            initial_latents = self.pipeline.text_encoder.encode(
                [text_prompt] * num_samples
            )
        return initial_latents

    def predict(self, x_t, timestep, text_prompt, condition_images):
        # Convert text prompt to conditioning embeddings
        text_embeddings = self.pipeline.text_encoder.encode([text_prompt])

        # Predict the noise using the ControlNet model
        noise_pred = self.pipeline.unet(
            x_t,
            timestep,
            encoder_hidden_states=text_embeddings,
            additional_inputs=condition_images,
        ).sample
        return noise_pred


class GLIGENPrior(Prior):
    def __init__(self, model_name="gligen/gligen", condition_parameters=None):
        super().__init__()
        self.pipeline = StableDiffusionGLIGENPipeline.from_pretrained(model_name).to(
            "cuda"
        )
        self.condition_parameters = condition_parameters

    def sample(self, text_prompt, num_samples=1):
        with torch.no_grad():
            initial_latents = self.pipeline.text_encoder.encode(
                [text_prompt] * num_samples
            )
        return initial_latents

    def predict(self, x_t, timestep, text_prompt, spatial_conditions):
        # Convert text prompt to conditioning embeddings
        text_embeddings = self.pipeline.text_encoder.encode([text_prompt])

        # Predict the noise using the GLIGEN model
        noise_pred = self.pipeline.unet(
            x_t,
            timestep,
            encoder_hidden_states=text_embeddings,
            additional_inputs=spatial_conditions,
        ).sample
        return noise_pred


class AniMaginePrior(Prior):
    def __init__(self):
        super().__init__()

        # Load VAE component
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )

        # Configure the pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-3.1",
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.to("cuda")

    def sample(self, text_prompt, num_samples=1, **kwargs):

        # set default kwargs
        if "negative_prompt" not in kwargs:
            kwargs["negative_prompt"] = [NEGATIVE_PROMPT]
        if "width" not in kwargs:
            kwargs["width"] = 832
        if "height" not in kwargs:
            kwargs["height"] = 1216
        if "guidance_scale" not in kwargs:
            kwargs["guidance_scale"] = 7
        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = 28

        kwargs["negative_prompt"] = kwargs["negative_prompt"] * num_samples

        with torch.no_grad():
            images = self.pipeline([text_prompt] * num_samples, **kwargs).images
        return images

    def predict(self, x_t, timestep, text_prompt):
        # Convert text prompt to conditioning embeddings
        text_embeddings = self.pipeline.text_encoder.encode([text_prompt])

        # Predict the noise using the UNet model
        noise_pred = self.pipeline.unet(
            x_t, timestep, encoder_hidden_states=text_embeddings
        ).sample
        return noise_pred
