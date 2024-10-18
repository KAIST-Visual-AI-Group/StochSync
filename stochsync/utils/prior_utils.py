import os
from numbers import Number
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
)

from .random_utils import seed_everything
from .image_utils import torch_to_pil, concat_images

def encode_text(pipeline, prompt, negative_prompt=None):
    """
    Encode a text prompt into a feature vector.
    """
    text_embeddings = pipeline.encode_prompt(
        prompt, "cuda", 1, True, negative_prompt=negative_prompt
    )
    # uncond, cond
    text_embeddings = torch.cat([text_embeddings[1], text_embeddings[0]])
    return text_embeddings

def encode_image(pipeline, img_tensor):
    vae = pipeline.vae
    flag = False
    if img_tensor.dim() == 3:
        flag = True
        img_tensor = img_tensor.unsqueeze(0)
    x = (2 * img_tensor - 1).to(vae.dtype)
    x = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    if flag:
        x = x.squeeze(0)
    return x

def decode_latent(pipeline, latent):
    vae = pipeline.vae
    flag = False
    if latent.dim() == 3:
        flag = True
        latent = latent.unsqueeze(0)
    x = vae.decode(latent / vae.config.scaling_factor).sample
    x = (x / 2 + 0.5).clamp(0, 1)
    if flag:
        x = x.squeeze(0)
    return x

def add_noise(pipeline, x, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x)
    noisy_sample = pipeline.scheduler.add_noise(x, noise, t)
    alpha_t = pipeline.scheduler.alphas_cumprod[t].to(x)
    beta_t = 1 - alpha_t
    noisy_sample_2 = alpha_t**0.5 * x + beta_t**0.5 * noise
    assert torch.allclose(noisy_sample, noisy_sample_2), f"{torch.max(torch.abs(noisy_sample - noisy_sample_2))}"
    return noisy_sample

def get_tweedie(pipeline, noisy_sample, eps_pred, t):
    alpha = pipeline.scheduler.alphas_cumprod[t].to(noisy_sample.device)
    tweedie = (noisy_sample - (1 - alpha) ** 0.5 * eps_pred) / alpha**0.5
    return tweedie

def get_eps(pipeline, noisy_sample, tweedie, t):
    alpha = pipeline.scheduler.alphas_cumprod[t]
    eps = (noisy_sample - (alpha**0.5) * tweedie) / (1 - alpha) ** 0.5
    return eps

def get_noisy_sample(pipeline, pred_original_sample, eps, t):
    alpha = pipeline.scheduler.alphas_cumprod[t].to(eps.device)
    noisy_sample = (alpha**0.5) * pred_original_sample + eps * (1 - alpha) ** 0.5
    return noisy_sample

def move_step(pipeline, sample, denoise_eps, src_t, tgt_t, renoise_eps=None):
    renoise_eps = renoise_eps if renoise_eps is not None else denoise_eps

    pred_original_sample = get_tweedie(pipeline, sample, denoise_eps, src_t)
    next_sample = get_noisy_sample(pipeline, pred_original_sample, renoise_eps, tgt_t)
    return next_sample

def get_pseudo_tweedie(pipeline, noisy_sample, eps_pred, t, lr=0.1):
    alpha = pipeline.scheduler.alphas_cumprod[t]
    tweedie = noisy_sample - lr * (1 - alpha) * eps_pred
    return tweedie

def get_pseudo_eps(pipeline, noisy_sample, tweedie, t, lr=0.1):
    alpha = pipeline.scheduler.alphas_cumprod[t]
    eps = (noisy_sample - tweedie) / (lr * (1 - alpha))
    return eps

def get_pseudo_noisy_sample(pipeline, pred_original_sample, eps, t, lr=0.1):
    alpha = pipeline.scheduler.alphas_cumprod[t]
    noisy_sample = pred_original_sample + eps * lr * (1 - alpha)
    return noisy_sample

def move_pseudo_step(pipeline, sample, denoise_eps, src_t, tgt_t, renoise_eps=None, lr=0.1, renoise_lr=None):
    renoise_eps = renoise_eps if renoise_eps is not None else denoise_eps
    renoise_lr = renoise_lr if renoise_lr is not None else lr

    pred_original_sample = get_pseudo_tweedie(pipeline, sample, denoise_eps, src_t, lr=lr)
    next_sample = get_pseudo_noisy_sample(pipeline, pred_original_sample, renoise_eps, tgt_t, lr=renoise_lr)
    return next_sample

def predict(pipeline, x_t, timestep, text_prompt, negative_prompt=None, guidance_scale=0.0):
    text_embeddings = encode_text(pipeline, text_prompt, negative_prompt)
    cond = {"encoder_hidden_states": text_embeddings}

    x_t_stack = torch.cat([x_t] * 2)
    noise_pred = pipeline.unet(x_t_stack, timestep, **cond).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return noise_pred
