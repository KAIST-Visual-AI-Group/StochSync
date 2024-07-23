from functools import lru_cache

from abc import ABC, abstractmethod
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionGLIGENPipeline,
)
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
)


#NEGATIVE_PROMPT = "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"
NEGATIVE_PROMPT = "deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke"


class Prior(ABC):
    def __init__(self):
        super().__init__()
        self.pipeline = None

    @abstractmethod
    def prepare_cond(self, camera):
        return None

    @abstractmethod
    def sample(self):
        """
        Generate images from a text prompt and other conditions.
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Predict the noise for a given latent `x_t` at a specific timestep.
        """
        pass

    @lru_cache(maxsize=10)
    def encode_text(self, prompt, negative_prompt=None):
        """
        Encode a text prompt into a feature vector.
        """
        assert self.pipeline is not None, "Pipeline not initialized"
        text_embeddings = self.pipeline.encode_prompt(
            prompt, "cuda", 1, True, negative_prompt=negative_prompt
        )
        # uncond, cond
        text_embeddings = torch.cat([text_embeddings[1], text_embeddings[0]])
        return text_embeddings

    def encode_image(self, img_tensor):
        assert self.pipeline is not None, "Pipeline not initialized"
        vae = self.pipeline.vae
        flag = False
        if img_tensor.dim() == 3:
            flag = True
            img_tensor = img_tensor.unsqueeze(0)
        x = (2 * img_tensor - 1).to(vae.dtype)
        x = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        if flag:
            x = x.squeeze(0)
        return x

    def decode_latent(self, latent):
        assert self.pipeline is not None, "Pipeline not initialized"
        vae = self.pipeline.vae
        flag = False
        if latent.dim() == 3:
            flag = True
            latent = latent.unsqueeze(0)
        x = vae.decode(latent / vae.config.scaling_factor).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        if flag:
            x = x.squeeze(0)
        return x

    def add_noise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        noisy_sample = self.pipeline.scheduler.add_noise(x, noise, t)
        alpha_t = self.pipeline.scheduler.alphas_cumprod[t].to(x)
        beta_t = 1 - alpha_t
        noisy_sample_2 = alpha_t**0.5 * x + beta_t**0.5 * noise
        assert torch.allclose(noisy_sample, noisy_sample_2), f"{torch.max(torch.abs(noisy_sample - noisy_sample_2))}"
        return noisy_sample
    
    def tweedie(self, x_t, eps_t, t):
        alpha_t = self.pipeline.scheduler.alphas_cumprod[t].to(x_t)
        beta_t = 1 - alpha_t
        x_0 = (x_t - beta_t**0.5 * eps_t) / alpha_t**0.5
        return x_0