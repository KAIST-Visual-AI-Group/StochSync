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
    
    # def tweedie(self, x_t, eps_t, t):
    #     alpha_t = self.pipeline.scheduler.alphas_cumprod[t].to(x_t)
    #     beta_t = 1 - alpha_t
    #     x_0 = (x_t - beta_t**0.5 * eps_t) / alpha_t**0.5
    #     return x_0

    def get_tweedie(self, noisy_sample, eps_pred, t):
        alpha = self.pipeline.scheduler.alphas_cumprod[t]
        tweedie = (noisy_sample - (1 - alpha) ** 0.5 * eps_pred) / alpha**0.5
        return tweedie

    def get_eps(self, noisy_sample, tweedie, t):
        alpha = self.pipeline.scheduler.alphas_cumprod[t]
        eps = (noisy_sample - (alpha**0.5) * tweedie) / (1 - alpha) ** 0.5
        return eps

    def get_noisy_sample(self, pred_original_sample, eps, t):
        alpha = self.pipeline.scheduler.alphas_cumprod[t]
        noisy_sample = (alpha**0.5) * pred_original_sample + eps * (1 - alpha) ** 0.5
        return noisy_sample

    def move_step(self, sample, denoise_eps, src_t, tgt_t, renoise_eps=None):
        renoise_eps = renoise_eps if renoise_eps is not None else denoise_eps

        pred_original_sample = self.get_tweedie(sample, denoise_eps, src_t)
        next_sample = self.get_noisy_sample(pred_original_sample, renoise_eps, tgt_t)
        return next_sample
    
    @torch.no_grad()
    def ddim_loop(self, camera, x_t, src_t, tgt_t, mode="cfg", guidance_scale=None):
        # make sure src_t is int (if tensor, convert to int)
        if isinstance(src_t, torch.Tensor):
            src_t = src_t.item()
        if isinstance(tgt_t, torch.Tensor):
            tgt_t = tgt_t.item()
        
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        x_t = x_t.detach()

        if src_t == tgt_t:
            return x_t
        elif src_t < tgt_t:
            timesteps = reversed(self.scheduler.timesteps)
            from_idx = torch.where(timesteps > src_t)[0]
            from_idx = from_idx[0] if len(from_idx) > 0 else len(timesteps)
            to_idx = torch.where(timesteps < tgt_t)[0]
            to_idx = to_idx[-1] if len(to_idx) > 0 else -1
            timesteps = torch.cat(
                [
                    torch.tensor([src_t]),
                    timesteps[from_idx : to_idx + 1],
                    torch.tensor([tgt_t]),
                ]
            )
        elif src_t > tgt_t:
            timesteps = self.scheduler.timesteps
            from_idx = torch.where(timesteps < src_t)[0]
            from_idx = from_idx[0] if len(from_idx) > 0 else len(timesteps)
            to_idx = torch.where(timesteps > tgt_t)[0]
            to_idx = to_idx[-1] if len(to_idx) > 0 else -1
            timesteps = torch.cat(
                [
                    torch.tensor([src_t]),
                    timesteps[from_idx : to_idx + 1],
                    torch.tensor([tgt_t]),
                ]
            )

        print(" ".join([str(i.item()) for i in timesteps]))

        for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
            noise_pred_dict = self.predict(
                camera, x_t, t_curr, guidance_scale=guidance_scale, return_dict=True
            )
            noise_pred, noise_pred_uncond, noise_pred_text = (
                noise_pred_dict["noise_pred"],
                noise_pred_dict["noise_pred_uncond"],
                noise_pred_dict["noise_pred_text"],
            )

            if mode == "cfg":
                renoise_eps = noise_pred
            elif mode == "sds":
                renoise_eps = torch.randn_like(noise_pred)
            elif mode == "cfg++":
                renoise_eps = noise_pred_uncond
            elif mode == "sdi":
                print(f"performing SDI at t=0 -> {t_next}")

                pred_original_sample = self.get_tweedie(x_t, noise_pred, t_curr)
                self.scheduler.set_timesteps(10)
                noisy_sample = self.ddim_loop(
                    camera,
                    pred_original_sample,
                    0,
                    t_curr,
                    guidance_scale=-guidance_scale,
                    mode="cfg",
                )
                self.scheduler.set_timesteps(30)
                noisy_sample = self.ddim_loop(
                    camera,
                    noisy_sample,
                    t_curr,
                    t_next,
                    guidance_scale=guidance_scale,
                    mode="cfg",
                )
                renoise_eps = self.get_eps(noisy_sample, pred_original_sample, t_next)

            x_t = self.move_step(
                x_t, noise_pred, t_curr, t_next, renoise_eps=renoise_eps
            )
        return x_t