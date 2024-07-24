import os
from random import randint
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .base import DistillationSampler
import shared_modules
from utils.extra_utils import ignore_kwargs

class SDSSampler(DistillationSampler):
    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        scale_factor: float = 1.0
        reduction: str = "sum"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

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
            noise_preds = prior.predict(camera, latent_noisy, t)

        w = 1 - prior.pipeline.scheduler.alphas_cumprod[t].to(latent)
        grad = w * (noise_preds - noise)
        target = (latent - grad).detach()
        if self.cfg.reduction == "mean":
            loss = 0.5 * F.mse_loss(latent, target, reduction="mean")
        else:
            loss = 0.5 * F.mse_loss(latent, target, reduction="sum") / latent.shape[0]

        return loss * self.cfg.scale_factor


class DreamTimeSampler(SDSSampler):
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


class SDISampler(SDSSampler):
    @ignore_kwargs
    @dataclass
    class Config:
        scale_factor: float = 1.0
        max_steps: int = 10000
        reduction: str = "sum"
        opt_steps: int = 0
        opt_lr: float = 0.01

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def sample_timestep(self, step):
        t_start = 960
        t_end = 200
        a = step / self.cfg.max_steps
        t = int(t_start + (t_end - t_start) * a)
        return torch.tensor(t, device="cuda")

    def sample_noise(self, camera, images, t):
        tau = randint(0, 33)
        t += tau

        ts_prev = len(shared_modules.prior.scheduler.timesteps)
        shared_modules.prior.scheduler.set_timesteps(10)
        noisy_sample = shared_modules.prior.ddim_loop(
            camera, images, 0, t, guidance_scale=-7.5
        )
        shared_modules.prior.scheduler.set_timesteps(ts_prev)

        alpha_prod_t = shared_modules.prior.scheduler.alphas_cumprod[t]
        inverted_eps = (noisy_sample - (alpha_prod_t**0.5) * images) / (
            1 - alpha_prod_t
        ) ** 0.5

        def fixed_point_loss(img, eps, t):
            assert img.dtype == eps.dtype
            data_dtype = img.dtype
            model_dtype = shared_modules.prior.pipeline.dtype
            noisy_sample = shared_modules.prior.get_noisy_sample(img, eps, t)
            noise_pred = shared_modules.prior.predict(camera, noisy_sample.to(model_dtype), t).to(data_dtype)
            return F.mse_loss(noise_pred, eps)
        
        if self.cfg.opt_steps > 0:
            print("Optimizing for fixed point")
            images = images.float().detach()
            inverted_eps = inverted_eps.float().detach().requires_grad_()
            opt = torch.optim.Adam([inverted_eps], lr=self.cfg.opt_lr)
            for _ in range(self.cfg.opt_steps):
                opt.zero_grad()
                loss = fixed_point_loss(images, inverted_eps, t)
                print(loss.item())
                loss.backward()
                opt.step()
            inverted_eps = inverted_eps.detach().to(shared_modules.prior.pipeline.dtype)

        h = 0.3 * (1 - alpha_prod_t) ** 0.5 * torch.randn_like(inverted_eps)

        return inverted_eps + h