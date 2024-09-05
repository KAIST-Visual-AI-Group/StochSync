from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.extra_utils import ignore_kwargs
import shared_modules as sm
from random import randint


class NoiseSampler(ABC):
    @ignore_kwargs
    @dataclass
    class Config:
        get_noise_from_model: bool = False

    def __init__(self, cfg):
        self.cfg = self.Config(**cfg)

    @abstractmethod
    def __call__(self, camera, images, t, *args, **kwargs):
        # Sample noise for the given camera, images, and timestep.
        pass

    def get_noise(self, camera, images):
        if self.cfg.get_noise_from_model:
            return sm.model.get_noise(camera)
        else:
            return torch.randn_like(images)


class SDSSampler(NoiseSampler):
    def __call__(self, camera, images, t, *args, **kwargs):
        return self.get_noise(camera, images)


class SDISampler(NoiseSampler):
    @ignore_kwargs
    @dataclass
    class Config(NoiseSampler.Config):
        inversion_guidance_scale: float = -7.5
        opt_steps: int = 0
        opt_lr: float = 0.01

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def __call__(self, camera, images, t, eps_prev, *args, **kwargs):
        if eps_prev is None:
            return self.get_noise(camera, images)
        
        tau = randint(0, 33)
        t_tau = min(t + tau, 999)

        ts_prev = len(sm.prior.scheduler.timesteps)
        sm.prior.scheduler.set_timesteps(10)
        noisy_sample = sm.prior.ddim_loop(
            camera,
            images,
            0,
            t_tau,
            guidance_scale=self.cfg.inversion_guidance_scale,
            mode="cfg",
        )
        sm.prior.scheduler.set_timesteps(ts_prev)

        alpha_prod_t = sm.prior.scheduler.alphas_cumprod[t_tau]
        inverted_eps = sm.prior.get_eps(noisy_sample, images, t_tau)

        def fixed_point_loss(img, eps, t):
            data_dtype = img.dtype
            model_dtype = sm.prior.dtype
            noisy_sample = sm.prior.get_noisy_sample(img, eps, t)
            noise_pred = sm.prior.predict(camera, noisy_sample.to(model_dtype), t).to(
                data_dtype
            )
            return F.mse_loss(noise_pred, eps)

        if self.cfg.opt_steps > 0:
            print("Optimizing for fixed point")
            images = images.float().detach()
            inverted_eps = inverted_eps.float().detach().requires_grad_()
            opt = torch.optim.Adam([inverted_eps], lr=self.cfg.opt_lr)
            for _ in range(self.cfg.opt_steps):
                loss = fixed_point_loss(images, inverted_eps, t_tau)
                loss.backward()
                opt.step()
                opt.zero_grad()
            inverted_eps = inverted_eps.detach().to(sm.prior.dtype)

        # h = 0.3 * (1 - alpha_prod_t) ** 0.5 * self.get_noise(camera, inverted_eps)
        # return inverted_eps + h
        return inverted_eps


class DDIMSampler(NoiseSampler):
    def __call__(self, camera, images, t, eps_prev, *args, **kwargs):
        if eps_prev is None:
            return self.get_noise(camera, images)
        return eps_prev


class RandomizedDDIMSampler(NoiseSampler):
    @ignore_kwargs
    @dataclass
    class Config(NoiseSampler.Config):
        random_noise_end: int = 560

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def __call__(self, camera, images, t, eps_prev, *args, **kwargs):
        if t >= self.cfg.random_noise_end or eps_prev == None:
            return self.get_noise(camera, images)
        return eps_prev
    
class RandomizedSDISampler(SDISampler):
    @ignore_kwargs
    @dataclass
    class Config(SDISampler.Config):
        random_noise_end: int = 560

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def __call__(self, camera, images, t, eps_prev, *args, **kwargs):
        if t >= self.cfg.random_noise_end or eps_prev == None:
            return self.get_noise(camera, images)
        
        tau = randint(0, 33)
        t_tau = min(t + tau, 999)

        ts_prev = len(sm.prior.scheduler.timesteps)
        sm.prior.scheduler.set_timesteps(10)
        noisy_sample = sm.prior.ddim_loop(
            camera,
            images,
            0,
            t_tau,
            guidance_scale=self.cfg.inversion_guidance_scale,
            mode="cfg",
        )
        sm.prior.scheduler.set_timesteps(ts_prev)

        alpha_prod_t = sm.prior.scheduler.alphas_cumprod[t_tau]
        inverted_eps = sm.prior.get_eps(noisy_sample, images, t_tau)

        def fixed_point_loss(img, eps, t):
            data_dtype = img.dtype
            model_dtype = sm.prior.dtype
            noisy_sample = sm.prior.get_noisy_sample(img, eps, t)
            noise_pred = sm.prior.predict(camera, noisy_sample.to(model_dtype), t).to(
                data_dtype
            )
            return F.mse_loss(noise_pred, eps)

        if self.cfg.opt_steps > 0:
            print("Optimizing for fixed point")
            images = images.float().detach()
            inverted_eps = inverted_eps.float().detach().requires_grad_()
            opt = torch.optim.Adam([inverted_eps], lr=self.cfg.opt_lr)
            for _ in range(self.cfg.opt_steps):
                loss = fixed_point_loss(images, inverted_eps, t_tau)
                loss.backward()
                opt.step()
                opt.zero_grad()
            inverted_eps = inverted_eps.detach().to(sm.prior.dtype)

        # h = 0.3 * (1 - alpha_prod_t) ** 0.5 * self.get_noise(camera, inverted_eps)
        # return inverted_eps + h
        return inverted_eps


class GeneralizedDDIMSampler(NoiseSampler):
    @ignore_kwargs
    @dataclass
    class Config(NoiseSampler.Config):
        random_ratio_expr: str = "0.5 * (1 - t / 1000)"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.random_ratio = eval(f"lambda t: {self.cfg.random_ratio_expr}")

    def __call__(self, camera, images, t, eps_prev, *args, **kwargs):
        random_eps = self.get_noise(camera, images)
        if eps_prev is None:
            return random_eps
        ratio = self.random_ratio(t)
        return ratio**0.5 * random_eps + (1 - ratio) ** 0.5 * eps_prev
