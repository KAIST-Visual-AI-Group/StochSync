import os
from random import randint
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .base import DistillationSampler
import shared_modules
from utils.extra_utils import ignore_kwargs
from utils.print_utils import print_warning
from utils.extra_utils import weak_lru


class CSDSampler(DistillationSampler):
    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        scale_factor: float = 1.0
        reduction: str = "sum"
        guidance_scale: float = 100.0

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.fixed_noise = torch.randn(1, 4, 64, 64, device="cuda")

    def sample_timestep(self, step):
        return torch.randint(20, 980, (), device="cuda")

    def sample_noise(self, camera, images, t):
        return torch.randn_like(images)

    def __call__(self, camera, images, step):
        prior = shared_modules.prior
        if images.shape[1] == 3:
            latent = prior.encode_image(images)
        else:
            latent = images

        # Encode latents
        t = self.sample_timestep(step)
        noise = self.sample_noise(camera, latent, t)
        latent_noisy = prior.add_noise(latent, t, noise=noise)

        # Calculate u-net loss and backprop
        with torch.no_grad():
            noise_preds = prior.predict(camera, latent_noisy, t, return_dict=True)

        w = 1 - prior.pipeline.scheduler.alphas_cumprod[t].to(latent)
        grad = w * (noise_preds["noise_pred_text"] - noise_preds["noise_pred_uncond"])
        target = (latent - grad).detach()
        if self.cfg.reduction == "mean":
            loss = 0.5 * F.mse_loss(latent, target, reduction="mean")
        else:
            loss = 0.5 * F.mse_loss(latent, target, reduction="sum") / latent.shape[0]

        return loss * self.cfg.scale_factor


class DreamTimeCSDSampler(CSDSampler):
    @ignore_kwargs
    @dataclass
    class Config:
        scale_factor: float = 1.0
        max_steps: int = 10000
        reduction: str = "sum"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def sample_timestep(self, step):
        t_start = 980
        t_end = 20
        a = step / self.cfg.max_steps
        t = int(t_start + (t_end - t_start) * a)
        return torch.tensor(t, device="cuda")


class BSDSampler(CSDSampler):
    @ignore_kwargs
    @dataclass
    class Config(CSDSampler.Config):
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        extra_tgt_prompts: str = ", detailed high resolution, high quality, sharp"
        extra_src_prompts: str = (
            ", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed"
        )

    def __init__(self, cfg):
        print_warning(
            "[Warning] This sampler incorporates text-engineering and is compatible only with StableDiffusionPrior."
        )
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def __call__(self, camera, images, step):
        prior = shared_modules.prior
        if images.shape[1] == 3:
            latent = prior.encode_image(images)
        else:
            latent = images

        # Encode latents
        t = self.sample_timestep(step)
        noise = self.sample_noise(camera, latent, t)
        latent_noisy = prior.add_noise(latent, t, noise=noise)

        # Calculate u-net loss and backprop
        text_prompt = self.cfg.text_prompt + self.cfg.extra_tgt_prompts
        negative_prompt = self.cfg.text_prompt + self.cfg.extra_src_prompts
        with torch.no_grad():
            noise_preds = prior.predict(
                camera,
                latent_noisy,
                t,
                return_dict=True,
                text_prompt=text_prompt,
                negative_prompt=negative_prompt,
            )

        w = 1 - prior.pipeline.scheduler.alphas_cumprod[t].to(latent)
        grad = w * (noise_preds["noise_pred_text"] - noise_preds["noise_pred_uncond"])
        # print their relative angle(cos value)
        print(
            torch.nn.functional.cosine_similarity(
                noise_preds["noise_pred_text"].flatten(),
                noise_preds["noise_pred_uncond"].flatten(),
                dim=0,
            ).item(),
            torch.nn.functional.cosine_similarity(
                noise.flatten(), noise_preds["noise_pred_text"].flatten(), dim=0
            ).item(),
        )
        # angle between noise snf noise_pred_text
        target = (latent - grad).detach()
        if self.cfg.reduction == "mean":
            loss = 0.5 * F.mse_loss(latent, target, reduction="mean")
        else:
            loss = 0.5 * F.mse_loss(latent, target, reduction="sum") / latent.shape[0]

        return loss * self.cfg.scale_factor


class BSDISampler(BSDSampler):
    @ignore_kwargs
    @dataclass
    class Config(CSDSampler.Config):
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        extra_tgt_prompts: str = ", detailed high resolution, high quality, sharp"
        extra_src_prompts: str = (
            ", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed"
        )
        opt_steps: int = 30
        opt_lr: float = 0.005
        inversion_guidance_scale: float = 0.0

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.cached_noise = None

    def sample_noise(self, camera, images, t):
        if self.cached_noise is not None:
            return self.cached_noise
        tau = randint(0, 33)
        t_tau = t + tau
        t_tau = min(max(t_tau, 0), 999)

        ts_prev = len(shared_modules.prior.scheduler.timesteps)
        shared_modules.prior.scheduler.set_timesteps(10)
        noisy_sample = shared_modules.prior.ddim_loop(
            camera,
            images,
            0,
            t_tau,
            guidance_scale=self.cfg.inversion_guidance_scale,
            mode="cfg",
        )
        shared_modules.prior.scheduler.set_timesteps(ts_prev)

        alpha_prod_t = shared_modules.prior.scheduler.alphas_cumprod[t_tau]
        inverted_eps = (noisy_sample - (alpha_prod_t**0.5) * images) / (
            1 - alpha_prod_t
        ) ** 0.5

        def fixed_point_loss(img, eps, t):
            assert img.dtype == eps.dtype
            data_dtype = img.dtype
            model_dtype = shared_modules.prior.pipeline.dtype
            noisy_sample = shared_modules.prior.get_noisy_sample(img, eps, t)
            noise_pred = shared_modules.prior.predict(
                camera, noisy_sample.to(model_dtype), t
            ).to(data_dtype)
            return F.mse_loss(noise_pred, eps)

        if self.cfg.opt_steps > 0:
            print("Optimizing for fixed point")
            images = images.float().detach()
            inverted_eps = inverted_eps.float().detach().requires_grad_()
            opt = torch.optim.Adam([inverted_eps], lr=self.cfg.opt_lr)
            for _ in range(self.cfg.opt_steps):
                opt.zero_grad()
                loss = fixed_point_loss(images, inverted_eps, t_tau)
                print(loss.item())
                loss.backward()
                opt.step()
            inverted_eps = inverted_eps.detach().to(shared_modules.prior.pipeline.dtype)

        h = 0.3 * (1 - alpha_prod_t) ** 0.5 * torch.randn_like(inverted_eps)
        self.cached_noise = inverted_eps
        return inverted_eps
