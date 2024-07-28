from typing import Dict, Literal
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
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers import (
    StableDiffusionXLPipeline,
    MarigoldDepthPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    DDIMScheduler,
)
#from third_party.mvdream_diffusers.mv_unet import get_camera, get_camera_specified
from third_party.mvdream.pipeline_mvdream import MVDreamPipeline

from .base import Prior, NEGATIVE_PROMPT
from utils.camera_utils import generate_camera_params, convert_camera_convention
from utils.extra_utils import (
    attach_direction_prompt,
    ignore_kwargs,
)  # TODO: How to remove this project-specific import?


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

    def predict(self, camera, x_t, timestep, guidance_scale=None, return_dict=False):
        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        self.prepare_cond(camera)
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


class ViewDependentStableDiffusionPrior(StableDiffusionPrior):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def prepare_cond(self, camera):
        text_prompts = attach_direction_prompt(
            self.cfg.text_prompt, camera["elevation"], camera["azimuth"]
        )

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

        batch_size: int = 1
        elevation: int = 0
        mixed_precision: bool = False

        convention: Literal[
            "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
        ] = "RDF"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

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

    def prepare_cond(self, camera, mv_camera):
        assert self.cfg.batch_size == len(camera["elevation"]) == len(mv_camera), (
            self.cfg.batch_size,
            len(camera["elevation"]),
            len(mv_camera),
        )

        text_prompts = attach_direction_prompt(
            self.cfg.text_prompt, camera["elevation"], camera["azimuth"]
        )

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
            "num_frames": self.cfg.batch_size,
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

    def predict(self, camera, x_t, timestep, guidance_scale=None, return_dict=False):
        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        # Get camera
        mv_camera = convert_camera_convention(camera["c2w"].cpu().numpy(), self.cfg.convention, "OpenGL")  # B 4 4
        #mv_camera = torch.tensor(mv_camera, dtype=torch.float32, device=x_t.device)
        mv_camera = torch.from_numpy(mv_camera).to(x_t.device)
        print(mv_camera.dtype)
        mv_camera[:, 0:3, 3] /= (mv_camera[:, 0:3, 3].norm(dim=-1, keepdim=True) + 1e-8)
        mv_camera = mv_camera.view(-1, 16)  # B 16
        #mv_camera = mv_camera.repeat_interleave(1, dim=0)
                                                
        self.prepare_cond(camera, mv_camera)
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        x_t_stack = torch.cat([x_t] * 2)

        timesteps = torch.stack([timestep] * self.cfg.batch_size * 2).to(x_t.device)
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
            model_name, variant="fp16", torch_dtype=torch.float16
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
