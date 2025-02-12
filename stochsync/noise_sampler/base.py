from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import randint
from math import sqrt, exp, log, cos, sin, pi, floor, ceil

import torch
import torch.nn.functional as F

from ..utils.extra_utils import ignore_kwargs
from ..utils.print_utils import print_warning, print_error, print_info
from .. import shared_modules as sm


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
        opt_lr: float = 0.01
        sdi_inv: bool = False  # Adding random noise at each step

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def __call__(self, camera, images, t, eps_prev, *args, **kwargs):
        
        tau = randint(0, 33)
        t_tau = min(t + tau, 999)
        
        noisy_sample = sm.prior.ddim_loop(
            camera,
            images,
            0,
            t_tau,
            guidance_scale=self.cfg.inversion_guidance_scale,
            mode="cfg",
            num_steps=10,
            sdi_inv=self.cfg.sdi_inv,
        )

        alpha_prod_t = sm.prior.ddim_scheduler.alphas_cumprod[t_tau]
        inverted_eps = sm.prior.get_eps(noisy_sample, images, t_tau)

        if self.cfg.sdi_inv:
            return inverted_eps
        
        h = 0.3 * (1 - alpha_prod_t) ** 0.5 * self.get_noise(camera, inverted_eps)
        return inverted_eps + h


class DDIMSampler(NoiseSampler):
    def __call__(self, camera, images, t, eps_prev, *args, **kwargs):
        if eps_prev is None:
            return self.get_noise(camera, images)
        return eps_prev


class GeneralizedDDIMSampler(NoiseSampler):
    @ignore_kwargs
    @dataclass
    class Config(NoiseSampler.Config):
        random_ratio_expr: str = "(1 - x)"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.random_ratio = lambda x: eval(str(self.cfg.random_ratio_expr))

    def __call__(self, camera, images, t, eps_prev, *args, **kwargs):
        random_eps = self.get_noise(camera, images)
        if eps_prev is None:
            return random_eps
        x = 1 - t / 1000
        ratio = self.random_ratio(x)
        # print(f"Stochasticiy ratio at t={t}: {ratio}")
        return (1 - ratio) ** 0.5 * eps_prev + ratio ** 0.5 * random_eps