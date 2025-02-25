import os
from os.path import join
import argparse
import math
from time import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler
from PIL import Image
from tqdm import tqdm

# --- Generator --- #

class StochSync(nn.Module):
    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-2-1-base",
        device="hpu",
        use_habana=True,
    ):
        super(StochSync, self).__init__()
        self.device = device
        self.model_id = model_id

        dtype = torch.float16

        if device == "hpu":
            # habana
            import habana_frameworks.torch.core as htcore
            # Enable hpu dynamic shape
            try:
                import habana_frameworks.torch.hpu as hthpu
                hthpu.enable_dynamic_shape()
            except ImportError:
                print("habana_frameworks could not be loaded")

            from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline

            self.pipe = GaudiStableDiffusionPipeline.from_pretrained(
                model_id,
                # scheduler=self.scheduler,
                use_habana=use_habana,
                use_hpu_graphs=True,
                gaudi_config="Habana/stable-diffusion",
                # torch_dtype=torch.bfloat16,
                torch_dtype=dtype,
            )
        else:
            from diffusers import StableDiffusionPipeline

            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).to(device)

        self.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Freeze models
        for p in self.pipe.unet.parameters():
            p.requires_grad_(False)
        for p in self.pipe.vae.parameters():
            p.requires_grad_(False)
        for p in self.pipe.text_encoder.parameters():
            p.requires_grad_(False)

        self.pipe.unet.eval() 
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        print(f"=== [INFO] SD successfully loaded!!! ===")

    @torch.no_grad()
    def forward(
        self,
        prompts="a photo of a dog",
        negative_prompts="",
        num_outer_steps=25,
        num_inner_steps=25,
        t_max=999,
        t_min=270,
        guidance_scale=7.5,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # encode prompt
        (
            prompt_embeds, 
            negative_prompt_embeds
        ) = self.pipe.encode_prompt(
            prompts,
            self.pipe._execution_device,
            1,
            True, # self.pipe.do_classifier_free_guidance,
            negative_prompts,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # define mapping parameters
        fov = 72
        phi = 0
        theta_group = [
            [0, 72, 144, 216, 288],
            [36, 108, 180, 252, 324],
        ]
        pers_height, pers_width = 512, 512
        pano_height, pano_width = 2048, 4096
        panorama = torch.zeros(1, 3, pano_height, pano_width).to(self.device)

        @torch.no_grad()
        def f(canonical_sample, it):
            # Projection
            thetas = theta_group[it % 2]
            img_projected = []
            for theta in thetas:
                img_projected.append(
                    pano_to_pers_raw(
                        canonical_sample,
                        fov,
                        theta,
                        phi,
                        pers_height,
                        pers_width,
                    )
                )
            img_projected = torch.cat(img_projected, dim=0)

            # VAE encode
            latent_projected = encode_image(self.pipe.vae, img_projected)

            return latent_projected
        
        @torch.no_grad()
        def g(instance_sample, it, original_canonical_sample=panorama):
            # VAE decode
            img_projected = decode_latent(self.pipe.vae, instance_sample)

            # Unprojection
            canonical_sample = torch.zeros(1, 3, pano_height, pano_width).to(self.device)
            canonical_count = torch.zeros(1, 1, pano_height, pano_width).to(self.device)
            thetas = theta_group[it % 2]

            for theta, img in zip(thetas, img_projected):
                pano, mask = pers_to_pano_raw(
                    img.unsqueeze(0),
                    fov,
                    theta,
                    phi,
                    pano_height,
                    pano_width,
                    return_mask=True,
                )
                canonical_sample += pano
                canonical_count += mask.view(1, 1, pano_height, pano_width)

            canonical_sample /= canonical_count + 1e-6

            if original_canonical_sample is not None:
                canonical_count = canonical_count.expand_as(original_canonical_sample)
                canonical_sample[canonical_count == 0] = original_canonical_sample[canonical_count == 0]

            return canonical_sample
        
        @torch.no_grad()
        def refine(instance_sample, t, seamless=False):
            raw_timesteps = torch.linspace(999, 0, num_inner_steps, dtype=torch.int32, device=self.device)
            
            from_idx = torch.where(raw_timesteps < t)[0]
            from_idx = from_idx[0] if len(from_idx) > 0 else len(raw_timesteps)
            to_idx = torch.where(raw_timesteps > 0)[0]
            to_idx = to_idx[-1] if len(to_idx) > 0 else -1
            timesteps = torch.cat(
                [
                    torch.tensor([t], device=self.device),
                    raw_timesteps[from_idx : to_idx + 1],
                    torch.tensor([0], device=self.device)
                ]
            )

            return sdedit(
                self.pipe.unet,
                self.scheduler,
                instance_sample,
                timesteps,
                prompt_embeds,
                guidance_scale=guidance_scale,
                seamless=seamless,
            )

        # set scheduler
        self.scheduler.set_timesteps(num_inner_steps)

        # t_max to t_min, excluding t_min, num_outer_steps elements.
        outer_timesteps = torch.linspace(t_max, t_min, num_outer_steps + 1, dtype=torch.int32, device=self.device)[:-1]

        # StochSync: Algorithm 4
        for i, t in enumerate(tqdm(outer_timesteps)):
            if i < len(theta_group):
                t = 999
            use_seamless = (i >= max(1, len(outer_timesteps) - 4))
            perspective = f(panorama, i)
            perspective = refine(perspective, t, seamless=use_seamless)
            panorama = g(perspective, i, original_canonical_sample=panorama)

        return torch_to_pil(panorama)
    
# --- Extra Utils --- #

def seed_everything(seed=42):
    """
    Seeds the random number generators of Python, Numpy and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def torch_to_pil(tensor, is_grayscale=False):
    # Convert a torch image tensor to a PIL image.
    # Input: tensor (HW or 1HW or 13HW or 3HW), is_grayscale (bool), cmap (str)
    # Output: PIL image

    if is_grayscale:
        assert tensor.dim() == 2 or (
            tensor.dim() == 3 and tensor.shape[0] == 1
        ), f"Grayscale tensor should be one of HW or 1HW: got {tensor.shape}."  # HW or 1HW
        # Make them all 3D tensor
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # 1HW
        tensor = tensor.repeat(3, 1, 1)
    else:
        assert (tensor.dim() == 3 and tensor.shape[0] == 3) or (
            tensor.dim() == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 3
        ), f"Color tensor should be one of 3HW or 13HW: got {tensor.shape}."  # 3HW or 13HW
        # Make them all 3D tensor
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

    tensor = tensor.clamp(0, 1)  # 3HW
    assert (
        tensor.dim() == 3 and tensor.shape[0] == 3
    ), f"Invalid tensor shape: {tensor.shape}"
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()
    tensor = (tensor * 255.0).astype(np.uint8)
    pil_tensor = Image.fromarray(tensor)
    return pil_tensor

# --- Encoder / Decoder / Forward / Backward --- #
def encode_image(vae, rgbs):
    flag = (rgbs.dim() == 3)
    rgbs = rgbs.unsqueeze(0) if flag else rgbs
    rgbs = rgbs.to(vae.dtype)
    
    latents = []
    for rgb in rgbs:
        tmp = 2 * rgb.unsqueeze(0) - 1
        tmp = vae.encode(tmp)[0].sample()
        tmp = tmp * vae.config.scaling_factor
        latents.append(tmp)
    latents = torch.cat(latents, dim=0)

    latents = latents.squeeze(0) if flag else latents
    return latents

def decode_latent(vae, latents):
    flag = (latents.dim() == 3)
    latents = latents.unsqueeze(0) if flag else latents
    latents = latents.to(vae.dtype)
    
    rgbs = []
    for latent in latents:
        tmp = latent.unsqueeze(0) / vae.config.scaling_factor
        tmp = vae.decode(tmp).sample
        tmp = (tmp / 2 + 0.5).clamp(0, 1)
        rgbs.append(tmp)
    rgbs = torch.cat(rgbs, dim=0)
    
    rgbs = rgbs.squeeze(0) if flag else rgbs
    return rgbs.to(torch.float32)

def get_tweedie(scheduler, noisy_sample, eps_pred, t):
    alpha = scheduler.alphas_cumprod[t]
    tweedie = (noisy_sample - (1 - alpha) ** 0.5 * eps_pred) / alpha**0.5
    return tweedie

def get_eps(scheduler, noisy_sample, tweedie, t):
    alpha = scheduler.alphas_cumprod[t]
    eps = (noisy_sample - (alpha**0.5) * tweedie) / (1 - alpha) ** 0.5
    return eps

def get_noisy_sample(scheduler, pred_original_sample, eps, t):
    alpha = scheduler.alphas_cumprod[t]
    noisy_sample = (alpha ** 0.5) * pred_original_sample + ((1 - alpha) ** 0.5) * eps
    return noisy_sample

def move_step(scheduler, sample, denoise_eps, src_t, tgt_t):
    pred_original_sample = get_tweedie(scheduler, sample, denoise_eps, src_t)
    next_sample = get_noisy_sample(scheduler, pred_original_sample, denoise_eps, tgt_t)
    return next_sample

@torch.no_grad()
def predict(unet, noisy_sample, t, prompt_embeds, guidance_scale=7.5):
    xt = torch.cat([noisy_sample] * 2)
    B = noisy_sample.shape[0]
    prompt_embeds = torch.repeat_interleave(prompt_embeds, B, dim=0)

    # print_info(f"xt shape: {xt.shape}, t: {t}, prompt_embeds shape: {prompt_embeds.shape}")
    noise_pred = unet(xt, t, encoder_hidden_states=prompt_embeds)['sample']
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    return noise_pred

@torch.no_grad()
def sdedit(unet, scheduler, sample, timesteps, prompt_embeds, guidance_scale=7.5, seamless=False):
    assert timesteps[-1] == 0, "The last timestep should be 0."

    T = timesteps[0]
    noise = torch.randn_like(sample)
    noisy_sample = None
    if T == 999:
        print_info("Forcing zero-SNR for T=999...")
        noisy_sample = noise
    else:
        noisy_sample = get_noisy_sample(scheduler, sample, noise, T)

    # Create the increasing and decreasing parts (each with 32 elements)
    increasing = torch.linspace(0, 1, steps=32)
    decreasing = torch.linspace(1, 0, steps=32)
    softmask = torch.cat([increasing, decreasing]).unsqueeze(0).repeat(64, 1).unsqueeze(0).unsqueeze(0).to(sample.device)

    N = len(timesteps)
    for i, (t_src, t_tgt) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        denoise_eps = predict(unet, noisy_sample, t_src, prompt_embeds, guidance_scale)
        noisy_sample = move_step(scheduler, noisy_sample, denoise_eps, t_src, t_tgt)
        if seamless:
            seamless_noisy_sample = get_noisy_sample(scheduler, sample, noise, t_tgt)
            seamless_mask = (softmask < (N - i) / N).to(noisy_sample.dtype)

            noisy_sample = seamless_mask * seamless_noisy_sample + (1 - seamless_mask) * noisy_sample

    return noisy_sample

# --- Print Utils --- #

class color:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    end = '\033[0m'

def print_with_box(text: str, box_color: str = color.purple, text_color: str = color.end, title: str = "", max_len = 88) -> None:
    """
    Prints a message with a box around it.
    """
    lines = text.split("\n")
    if len(title) > max_len - 3:
        title = title[:max_len - 6] + "..."
    text_len = max([len(line) for line in lines])
    title_len = len(title)
    line_len = min(max_len, max(title_len, text_len))

    # if each line is longer than max_len, break it into multiple lines
    new_lines = []
    for line in lines:
        while len(line) > line_len:
            new_lines.append(line[:line_len])
            line = line[line_len:]
        new_lines.append(line)
    lines = new_lines

    bar_len = line_len - len(title)
    front_bar_len = bar_len // 2
    back_bar_len = bar_len - front_bar_len
    print(box_color+"╭─" + "─"*front_bar_len + title + "─"*back_bar_len + "─╮"+color.end)
    for line in lines:
        print(box_color+"│ " + text_color + line.ljust(line_len) + box_color + " │"+color.end)
    print(box_color+"╰" + "─" * (line_len + 2) + "╯"+color.end)

def print_warning(*args) -> None:
    text = ' '.join(map(str, args))
    print(color.yellow + color.bold + '[Warning] ' + color.end + color.yellow + text + color.end)

def print_info(*args) -> None:
    text = ' '.join(map(str, args))
    print(color.green + color.bold + '[Info] ' + color.end + color.green + text + color.end)

def print_error(*args) -> None:
    text = ' '.join(map(str, args))
    print(color.red + color.bold + '[Error] ' + color.end + color.red + text + color.end)

# --- Mapping Functions --- #

def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: A tensor representing a quaternion (w, x, y, z).

    Returns:
        A tensor representing the corresponding rotation matrix.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return torch.Tensor(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )

def normalize_grid(grid, height, width):
    """
    Normalize map coordinates to the range [-1, 1] for use with grid_sample.
    """
    grid = grid.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0
    return grid

def gather_2d(tensor, grid):
    # tensor: BCHW -> BCX
    # grid: Bhw2 -> BHW -> B1
    # output: BChw
    B, C, H, W = tensor.shape
    b, h, w, _ = grid.shape
    assert b == B, f"Batch size mismatch: {b} != {B}"

    tensor = tensor.contiguous()
    lin_idx = (
        (grid[..., 0] * tensor.shape[-1] + grid[..., 1])
        .view(B, 1, -1)
        .expand(-1, C, -1)
    )
    tensor = tensor.view(B, C, tensor.shape[-2] * tensor.shape[-1])
    return tensor.gather(-1, lin_idx).view(B, C, h, w)

def remap(image, grid, mode="bilinear", padding_mode="border"):
    """
    Remap an image based on provided coordinate maps using grid_sample.
    """
    B, C, H, W = image.shape
    grid = normalize_grid(grid, H, W)
    grid = grid.unsqueeze(0) if grid.dim() == 3 else grid
    remapped_image = F.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )
    return remapped_image

def remap_int(tensor, grid, indexing="xy"):
    H, W = tensor.shape[-2:]
    if indexing == "xy":
        grid = grid.flip(-1)

    grid = grid.unsqueeze(0) if grid.dim() == 3 else grid
    grid = grid.long()
    grid[..., 0] = grid[..., 0].clamp(0, H - 2)
    grid[..., 1] = grid[..., 1].clamp(0, W - 2)
    return gather_2d(tensor, grid)

def xyz_to_lonlat(xyz):
    """
    Convert XYZ coordinates to longitude and latitude.
    """
    norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    xyz_norm = xyz / norm
    x, y, z = xyz_norm[..., 0], xyz_norm[..., 1], xyz_norm[..., 2]

    lon = torch.atan2(x, z)
    lat = torch.asin(y)

    return torch.stack([lon, lat], dim=-1)


def lonlat_to_xyz(lonlat):
    """
    Convert longitude and latitude to XYZ coordinates.
    """
    lon, lat = lonlat[..., 0], lonlat[..., 1]

    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)

    return torch.stack([x, y, z], dim=-1)


def project_xyz(xyz, K, eps=1e-6):
    """
    Project XYZ coordinates to the image plane using the intrinsic matrix K.
    """
    valid_mask = xyz[..., 2] > eps
    xyz_proj = xyz.clone()
    xyz_proj[~valid_mask] = float("inf")
    xyz_proj[valid_mask] /= xyz_proj[valid_mask][..., 2:3]
    xyz_proj[valid_mask] = torch.matmul(xyz_proj[valid_mask], K.T)
    return xyz_proj[..., :2]


def compute_intrinsic_matrix(fov, width, height):
    """
    Compute the camera intrinsic matrix based on FOV and image dimensions.
    """
    f = 0.5 * height / math.tan(0.5 * fov * math.pi / 180)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=torch.float32)


def rodriques(v):
    """
    Compute the Rodrigues rotation matrix.
    """
    theta = torch.linalg.norm(v)
    if theta == 0:
        return torch.eye(3)
    v = v / theta
    K = torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=torch.float32
    )
    return torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def rotation_matrix(theta, phi, roll=0):
    """
    Compute the rotation matrix based on theta and phi angles.
    """
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    R0 = rodriques(z_axis * math.radians(roll))
    R1 = rodriques(x_axis * math.radians(phi))
    R2 = rodriques(y_axis * math.radians(theta))

    return R2 @ R1 @ R0


def lonlat_to_xy(lonlat, height, width, lonrange=(-math.pi, math.pi), latrange=(-math.pi / 2, math.pi / 2)):
    """
    Convert longitude and latitude to pixel coordinates.
    """
    nlon = (lonlat[..., 0] - lonrange[0]) / (lonrange[1] - lonrange[0])
    nlat = (lonlat[..., 1] - latrange[0]) / (latrange[1] - latrange[0])
    X = nlon * width
    Y = nlat * height
    return torch.stack([X, Y], dim=-1)

def xy_to_lonlat(xy, height, width, lonrange=(-math.pi, math.pi), latrange=(-math.pi / 2, math.pi / 2)):
    """
    Convert pixel coordinates to longitude and latitude.
    """
    nx = xy[..., 0] / width
    ny = xy[..., 1] / height
    lon = nx * (lonrange[1] - lonrange[0]) + lonrange[0]
    lat = ny * (latrange[1] - latrange[0]) + latrange[0]
    return torch.stack([lon, lat], dim=-1)

@lru_cache(maxsize=32)
def compute_pano2pers_map(
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    pano_height,
    pano_width,
    quat=None,
    lonlat_to_xy=lonlat_to_xy,
):
    """
    Compute the XY mapping from the panorama to perspective view.
    """
    K = compute_intrinsic_matrix(fov, pers_width, pers_height)
    K_inv = torch.inverse(K)

    x, y = torch.meshgrid(
        torch.arange(pers_width), torch.arange(pers_height), indexing="xy"
    )
    xyz = torch.stack([x.float(), y.float(), torch.ones_like(x).float()], dim=-1)
    xyz = torch.matmul(xyz, K_inv.T)

    R_base = torch.eye(3)
    if quat is not None:
        R_base = quat_to_rot(quat)
    
    R = R_base @ rotation_matrix(theta, phi)
    xyz = torch.matmul(xyz, R.T)

    lonlat = xyz_to_lonlat(xyz)
    return lonlat_to_xy(lonlat, pano_height, pano_width)


@lru_cache(maxsize=32)
def compute_pers2pano_map(
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    pano_height,
    pano_width,
    quat=None,
    xy_to_lonlat=xy_to_lonlat,
):
    """
    Compute the inverse XY mapping from the perspective view to panorama.
    """
    K = compute_intrinsic_matrix(fov, pers_width, pers_height)

    x, y = torch.meshgrid(
        torch.arange(pano_width), torch.arange(pano_height), indexing="xy"
    )
    lonlat = xy_to_lonlat(torch.stack([x, y], dim=-1), pano_height, pano_width)

    R_base = torch.eye(3)
    if quat is not None:
        R_base = quat_to_rot(quat)
    
    R = R_base @ rotation_matrix(theta, phi)
    xyz = lonlat_to_xyz(lonlat)
    xyz = torch.matmul(xyz, R)

    return project_xyz(xyz, K)

def pano_to_pers(panorama, pano2pers, mode="nearest"):
    perspective_image = remap(panorama, pano2pers, mode)
    return perspective_image


def pano_to_pers_raw(
    panorama,
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    mode="nearest",
    mapping_func = compute_pano2pers_map,
    **kwargs
):
    """
    Transform a panorama image to a perspective view.
    """
    pano_height, pano_width = panorama.shape[-2], panorama.shape[-1]
    pano2pers = mapping_func(
        fov, theta, phi, pers_height, pers_width, pano_height, pano_width, **kwargs
    )
    pano2pers = pano2pers.to(panorama.device)
    return pano_to_pers(panorama, pano2pers, mode)


def pers_to_pano(perspective, pers2pano, return_mask=False, mode="nearest"):
    pers_height, pers_width = perspective.shape[-2], perspective.shape[-1]
    safe_padding = 1.0
    valid_mask = (
        (pers2pano[..., 0] >= -0.5 - safe_padding)
        & (pers2pano[..., 0] < pers_width - 0.5 + safe_padding)
        & (pers2pano[..., 1] >= -0.5 - safe_padding)
        & (pers2pano[..., 1] < pers_height - 0.5 + safe_padding)
    )

    if perspective.dtype == torch.float32:
        panorama_image = remap(perspective, pers2pano, mode) * valid_mask
    else:
        panorama_image = remap_int(perspective, pers2pano.round().long()) * valid_mask

    if return_mask:
        return panorama_image, valid_mask
    return panorama_image


def pers_to_pano_raw(
    perspective,
    fov,
    theta,
    phi,
    pano_height,
    pano_width,
    return_mask=False,
    mode="nearest",
    mapping_func = compute_pers2pano_map,
    **kwargs
):
    """
    Transform a perspective view image to a panorama.
    """
    pers_height, pers_width = perspective.shape[-2], perspective.shape[-1]
    pers2pano = mapping_func(
        fov, theta, phi, pers_height, pers_width, pano_height, pano_width, **kwargs
    )
    pers2pano = pers2pano.to(perspective.device)
    return pers_to_pano(perspective, pers2pano, return_mask, mode)


# --- Main --- #

if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--prompt", type=str, default="Graffiti-covered alleyway on a sunny afternoon")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, bad anatomy, disfigured, poorly drawn face")
    parser.add_argument("--num_outer_steps", type=int, default=25)
    parser.add_argument("--num_inner_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--t_max", type=int, default=999)
    parser.add_argument("--t_min", type=int, default=270)
    parser.add_argument("--save_dir", type=str, default="./outputs/stochsync")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_name", type=str, default="compute_time.txt")
    # check the type of compute
    parser.add_argument("--compute", type=str, default="hpu")
    args = parser.parse_args()

    if args.compute == "hpu":
        # habana
        import habana_frameworks.torch.core as htcore
        # Enable hpu dynamic shape
        try:
            import habana_frameworks.torch.hpu as hthpu
            hthpu.enable_dynamic_shape()
        except ImportError:
            print("habana_frameworks could not be loaded")

    # load model
    model = StochSync(model_id=args.model_id, device=args.compute, use_habana=True)

    # make save directory
    prompt_name = args.prompt.replace(" ", "_")
    args.save_dir = join(args.save_dir, prompt_name)
    os.makedirs(args.save_dir, exist_ok=True)

    inference_time_list = []

    # inference
    for i in range(args.num_images + 1):
        seed = args.seed + i
        seed_everything(seed)

        # measure time
        start_time = time()
        
        image = model(
            prompts=args.prompt,
            negative_prompts=args.negative_prompt,
            num_outer_steps=args.num_outer_steps,
            num_inner_steps=args.num_inner_steps,
            t_max=args.t_max,
            t_min=args.t_min,
            guidance_scale=args.guidance_scale,
        )

        end_time = time()

        image.save(join(args.save_dir, f"image_{seed:03d}.png"))
        print_info(f"Saved image_{seed:03d}.png")

        inference_time = end_time - start_time
        print_info(f"Inference time: {inference_time:.2f} seconds")

        if i > 0: # skip the first image (first inference time is usually longer. why?)
            inference_time_list.append(inference_time)

    # save inference time
    avg_inference_time = np.mean(inference_time_list)

    with open(os.path.join(args.save_dir, args.save_name), "w") as f:
        f.write(f"Average inference time: {avg_inference_time:.2f} seconds\n")
        f.write("Inference time list:\n")
        for i, _time in enumerate(inference_time_list):
            f.write(f"Image {i+1}: {_time:.2f} seconds\n")
        f.write("\n")