from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt, exp, log, cos, sin, pi, floor, ceil

import torch
import torch.nn.functional as F

from utils.extra_utils import ignore_kwargs
import shared_modules as sm
from random import randint
from utils.print_utils import print_warning, print_error, print_info


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
            print_warning(f"No previous noise found, generating random noise. This should only happen at the first step {t}")
            return self.get_noise(camera, images)
        
        tau = randint(0, 33)
        t_tau = min(t + tau, 999)

        # ========================================
        sdi_inv = False
        # Adding random noise at each step
        # ========================================
        
        noisy_sample = sm.prior.ddim_loop(
            camera,
            images,
            0,
            t_tau,
            guidance_scale=self.cfg.inversion_guidance_scale,
            mode="cfg",
            num_steps=10,
            sdi_inv=sdi_inv, 
        )

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
            raise NotImplementedError("Optimization for fixed point is not implemented")
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

        if sdi_inv:
            return inverted_eps
        
        h = 0.3 * (1 - alpha_prod_t) ** 0.5 * self.get_noise(camera, inverted_eps)
        return inverted_eps + h
        


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
        # if not hasattr(self, 'preparing_t'):
        #     print_warning("RandomizedDDIM is looking for the DOOM's day...")
        #     print_warning("DOOM's day is coming...")
        #     for step in range(100000):
        #         tmp = sm.time_sampler(step)
        #         if tmp < self.cfg.random_noise_end:
        #             self.preparing_t = sm.time_sampler(step - 2)
        #             print_warning(f"You must prepare for the DOOM's day at {self.preparing_t}")
        #             break
        #     else:
        #         raise ValueError("DOOM's day is a lie lol")
        # if t <= self.preparing_t:
        #     self.preparing_t = -666
        #     print_error("DOOM's day is here!")
        #     print_warning("The ground trembles beneath your feet as the air grows thick with an ominous presence.")
        #     print_warning("A dark cloud looms overhead, casting a shadow over the land.")
        #     print_warning("Whispers of ancient prophecies fill your ears, foretelling the coming of DOOM's day.")
        #     print_warning("Suddenly, the ground splits open, revealing a swirling vortex of darkness.")
        #     print_error("The hellgate has opened!")
        #     print_warning("Monstrous creatures emerge from the abyss, their eyes glowing with malevolence.")
        #     print_warning("You draw your weapon, ready to face the horrors that await.")
        #     print_warning("The fate of the world rests on your shoulders.")
        #     assert sm.dataset.__class__.__name__ == "RandomMVCameraDataset", "DOOM's day is only for RandomMVCameraDataset"
        #     sm.dataset.cfg.batch_size = 10
        #     sm.dataset.cfg.azim_range = (0, 0)

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
        if not hasattr(self, 'preparing_t'):
            print_warning("RandomizedSDI is looking for the DOOM's day...")
            print_warning("DOOM's day is coming...")
            for step in range(100000):
                tmp = sm.time_sampler(step)
                if tmp < self.cfg.random_noise_end:
                    self.preparing_t = sm.time_sampler(step - 2)
                    print_warning(f"You must prepare for the DOOM's day at {self.preparing_t}")
                    break
            else:
                raise ValueError("DOOM's day is a lie lol")
        if t <= self.preparing_t:
            self.preparing_t = -666
            print_error("DOOM's day is here!")
            print_warning("The ground trembles beneath your feet as the air grows thick with an ominous presence.")
            print_warning("A dark cloud looms overhead, casting a shadow over the land.")
            print_warning("Whispers of ancient prophecies fill your ears, foretelling the coming of DOOM's day.")
            print_warning("Suddenly, the ground splits open, revealing a swirling vortex of darkness.")
            print_error("The hellgate has opened!")
            print_warning("Monstrous creatures emerge from the abyss, their eyes glowing with malevolence.")
            print_warning("You draw your weapon, ready to face the horrors that await.")
            print_warning("The fate of the world rests on your shoulders.")
            assert sm.dataset.__class__.__name__ == "RandomMVCameraDataset", "DOOM's day is only for RandomMVCameraDataset"
            sm.dataset.cfg.batch_size = 10
            sm.dataset.cfg.azim_range = (0, 0)

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

        h = 0.3 * (1 - alpha_prod_t) ** 0.5 * self.get_noise(camera, inverted_eps)
        return inverted_eps + h
        # return inverted_eps


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


# ====================================================================
# ======================= Distillation-based =========================
# ====================================================================

class ISMSampler(NoiseSampler):
    @ignore_kwargs
    @dataclass
    class Config(NoiseSampler.Config):
        inversion_guidance_scale: float = 0.0
        interval_time: int = 50
        inv_steps: int = 50

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def __call__(self, camera, images, tgt_t, eps_prev=None, *args, **kwargs):
        src_t = tgt_t - 50
        
        num_steps = src_t // self.cfg.inv_steps
        with torch.no_grad():
            # Sample x_s with unconditional prompt embeddings 
            src_noisy_sample = sm.prior.ddim_loop(
                camera,
                images,
                0,
                src_t,
                guidance_scale=self.cfg.inversion_guidance_scale, 
                mode="cfg",
                num_steps=num_steps,
                sdi_inv=False, 
            )
            # Predict noise from x_s and x_t 
            src_noise_pred = sm.prior.predict(
                camera=camera, 
                x_t=src_noisy_sample, 
                timestep=src_t,
                return_dict=True,
            )["noise_pred_uncond"]

            # Sample x_t from x_s with unconditional prompt embeddings
            tgt_noisy_sample = sm.prior.move_step(
                src_noisy_sample, 
                src_noise_pred, 
                src_t, 
                tgt_t, 
                eta=0,
            )
            
            tgt_noise_pred = sm.prior.predict(
                camera=camera, 
                x_t=tgt_noisy_sample, 
                timestep=tgt_t,
                guidance_scale=7.5,
                return_dict=True,
            )["noise_pred"]
        
        alpha_t = sm.prior.pipeline.scheduler.alphas_cumprod.to(tgt_noisy_sample)[tgt_t]
        coeff = ((1 - alpha_t) * alpha_t) ** 0.5
        
        grad = coeff * (tgt_noise_pred - src_noise_pred)
        grad = torch.nan_to_num(grad)
        
        targets = (images - grad).detach()
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]
        
        return loss 
        