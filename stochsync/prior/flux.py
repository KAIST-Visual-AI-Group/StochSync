from dataclasses import dataclass
import inspect

import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers import ControlNetModel
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
import numpy as np
import cv2

from ..utils.camera_utils import convert_camera_convention, camera_hash
from ..utils.extra_utils import (
    ignore_kwargs,
    attach_direction_prompt,
    attach_detailed_direction_prompt,
    attach_elevation_prompt,
)
from ..utils.print_utils import print_info, print_warning, print_error
from .. import shared_modules as sm

from .base import Prior, NEGATIVE_PROMPT


def preprocess_depth(depth, mask):
    disp = depth.clone()
    disp[mask] = 1 / (disp[mask] + 1e-15)
    _shape = disp.shape
    if mask.dim() == 3:
        mask = mask

    B = _shape[0]
    disp_flat = disp.view(B, -1)
    mask_flat = mask.view(B, -1)

    inf_tensor = torch.full_like(disp_flat, float("inf"))
    neg_inf_tensor = torch.full_like(disp_flat, float("-inf"))

    _min = torch.where(mask_flat, disp_flat, inf_tensor).min(dim=1).values
    _max = torch.where(mask_flat, disp_flat, neg_inf_tensor).max(dim=1).values

    disp = (disp - _min.view(-1, 1, 1, 1)) / ((_max - _min + 1e-7).view(-1, 1, 1, 1))
    disp[~mask] = 0
    return disp


def canny_edge_detection(img, threshold1=100, threshold2=200):
    img_np = img.cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    edges = []
    # edges = cv2.Canny(img_np, threshold1, threshold2)
    for i in range(img_np.shape[0]):
        edges.append(cv2.Canny(img_np[i], threshold1, threshold2))
    edges = np.stack(edges, axis=0) / 255
    return torch.tensor(edges, dtype=img.dtype, device=img.device)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps = None,
    device = None,
    timesteps = None,
    sigmas = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class FluxDepthCannyPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        model_name: str = "stabilityai/stable-diffusion-2-depth"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = NEGATIVE_PROMPT
        width: int = 512
        height: int = 512
        guidance_scale: int = 100
        mixed_precision: bool = False
        root_dir: str = "./results/default"
        use_view_dependent_prompt: bool = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        if not (
            (self.cfg.width == self.cfg.height == 768)
            or (self.cfg.width == self.cfg.height == 96)
        ):
            print_error("Width and height must be 768(96) for Stable Diffusion")
            raise ValueError

        base_model = "black-forest-labs/FLUX.1-dev"
        controlnet_model_union = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"

        controlnet_union = FluxControlNetModel.from_pretrained(
            controlnet_model_union, torch_dtype=torch.bfloat16
        )
        controlnet = FluxMultiControlNetModel(
            [controlnet_union, controlnet_union]
        )  # we always recommend loading via FluxMultiControlNetModel

        self.ddim_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model, subfolder="scheduler"
        )

        self.pipeline = FluxControlNetPipeline.from_pretrained(
            base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
        ).to("cuda")

        mu = calculate_shift(
            16,
            self.pipeline.scheduler.config.base_image_seq_len,
            self.pipeline.scheduler.config.max_image_seq_len,
            self.pipeline.scheduler.config.base_shift,
            self.pipeline.scheduler.config.max_shift,
        )

        self.ddim_scheduler.set_timesteps(20, mu=mu)
        self.pipeline.transformer.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.controlnet.requires_grad_(False)
        self.pipeline.unet = self.pipeline.transformer  # for compatibility

        self.depth = None
        self.control_images = None
        self.prev_camera_hash = -65535

    @property
    def rgb_res(self):
        return 1, 3, 768, 768

    @property
    def latent_res(self):
        return 1, 16, 96, 96
    
    def encode_image(self, img_tensor):
        assert self.pipeline is not None, "Pipeline not initialized"
        vae = self.pipeline.vae
        flag = False
        if img_tensor.dim() == 3:
            flag = True
            img_tensor = img_tensor.unsqueeze(0)
        # image = img_tensor.to(vae.dtype)
        image = (2 * img_tensor - 1).to(vae.dtype)
        
        x = []
        for i in range(len(image)):
            y = vae.encode(image[i:i+1]).latent_dist.sample()
            y = (y - vae.config.shift_factor) * vae.config.scaling_factor
            x.append(y)
            
        x = torch.cat(x, dim=0)
        # x = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
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
        
        latent = latent.to(vae.dtype)
        
        x = []
        for i in range(len(latent)):
            y = (latent[i:i+1] / vae.config.scaling_factor) + vae.config.shift_factor
            y = vae.decode(y, return_dict=False)[0]
            y = sm.prior.pipeline.image_processor.postprocess(y, output_type='pt')
            x.append(y)
        x = torch.cat(x, dim=0)
        
        x = (x).clamp(0, 1)
        if flag:
            x = x.squeeze(0)
        return x.to(torch.float32)
    
    def add_noise(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        # alpha_t = self.ddim_scheduler.alphas_cumprod[t].to(x)
        # noisy_sample = alpha_t**0.5 * x + (1 - alpha_t)**0.5 * noise
        t = t / 1000
        print(t)
        noisy_sample = (1 - t) * x + t * noise

        return noisy_sample

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt

        if self.cfg.use_view_dependent_prompt:
            text_prompts = attach_detailed_direction_prompt(
                text_prompt, camera["elevation"], camera["azimuth"]
            )
        else:
            text_prompts = [text_prompt] * camera["num"]

        prompt_embeds = []
        pooled_prompt_embeds = []
        text_ids = []
        for prompt in text_prompts:
            print(prompt)
            (
                embeds,
                pooled_embeds,
                _text_ids,
            ) = self.pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
            )
            prompt_embeds.append(embeds)
            pooled_prompt_embeds.append(pooled_embeds)
            text_ids.append(_text_ids)
        prompt_embeds = torch.cat(prompt_embeds, dim=0)
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=0)
        # print("prompts", prompt_embeds.shape, pooled_prompt_embeds.shape)
        text_ids = text_ids[0]

        batch_size = camera["num"]
        height = self.cfg.height
        width = self.cfg.width
        latent_image_ids = self.pipeline._prepare_latent_image_ids(
            batch_size,
            2 * (int(height) // self.pipeline.vae_scale_factor),
            2 * (int(width) // self.pipeline.vae_scale_factor),
            device=self.device,
            dtype=torch.bfloat16,
        )

        self.cond = {
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
        }

        cam_hash = camera_hash(camera)
        if cam_hash != self.prev_camera_hash:
            # Render depth
            pkg = sm.model.render(camera, bsdf="depth", filter_mode="nearest")
            depth, mask = pkg["image"][:, :1], pkg["alpha"] > 0.99
            disp = preprocess_depth(depth, mask)
            disp = disp.expand(-1, 3, -1, -1).to(torch.bfloat16)

            # Render normal and canny edge
            pkg = sm.model.render(camera, bsdf="normal", filter_mode="nearest")
            mask = pkg["alpha"][:, 0] > 0.99
            normal = (2 * pkg["image"] - 1).permute(0, 2, 3, 1) @ camera["c2w"][
                :, None, :3, :3
            ]
            normal = (normal.permute(0, 3, 1, 2) + 1) / 2
            normal[:, 0][~mask] = 0.5
            normal[:, 1][~mask] = 0.5
            normal[:, 2][~mask] = 1.0
            canny = canny_edge_detection(normal.permute(0, 2, 3, 1))
            canny = canny.unsqueeze(1).expand(-1, 3, -1, -1).to(torch.bfloat16)

            # Encode depth
            disp = self.encode_image(disp)
            height_disp, width_disp = disp.shape[2:]
            disp = self.pipeline._pack_latents(
                disp,
                camera["num"],
                self.pipeline.transformer.config.in_channels // 4,
                height_disp,
                width_disp,
            )

            # Encode canny
            canny = self.encode_image(canny)
            height_canny, width_canny = canny.shape[2:]
            canny = self.pipeline._pack_latents(
                canny,
                camera["num"],
                self.pipeline.transformer.config.in_channels // 4,
                height_canny,
                width_canny,
            )
            # print(f"disp: {disp.shape}, canny: {canny.shape}")
            self.control_images = [disp, canny]

            self.prev_camera_hash = cam_hash

        return self.cond

    def sample(self, text_prompt, num_samples=1):
        raise NotImplementedError

    def fast_sample(
        self,
        camera,
        x_t,
        timesteps,
        guidance_scale=None,
        text_prompt=None,
        negative_prompt=None,
        conditioning_scale=[0.35, 0.25],
    ):
        # num_inference_steps = len(timesteps)
        # B, C, H, W = x_t.shape
        # image_seq_len = self.pipeline._pack_latents(x_t, B, C, H, W).shape[1]
        # mu = calculate_shift(
        #     image_seq_len,
        #     self.pipeline.scheduler.config.base_image_seq_len,
        #     self.pipeline.scheduler.config.max_image_seq_len,
        #     self.pipeline.scheduler.config.base_shift,
        #     self.pipeline.scheduler.config.max_shift,
        # )
        # sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        # timesteps, num_inference_steps = retrieve_timesteps(
        #     self.pipeline.scheduler,
        #     num_inference_steps,
        #     self.device,
        #     # timesteps,
        #     sigmas=sigmas,
        #     mu=mu,
        # )

        # self.ddim_scheduler.set_timesteps(num_inference_steps, mu=mu)
        # self.ddim_scheduler.set_timesteps(timesteps=timesteps)
        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            interval = (t_curr - t_next) / 1000
            noise_pred = self.predict(camera, x_t, t_curr, guidance_scale, text_prompt=text_prompt, conditioning_scale=conditioning_scale)
            x_t = x_t - interval * noise_pred

        return x_t

    @torch.no_grad()
    def predict(
        self,
        camera,
        x_t,
        timestep,
        guidance_scale=None,
        return_dict=False,
        text_prompt=None,
        negative_prompt=None,
        conditioning_scale=[0.35, 0.25],
    ):
        # Predict the noise using the UNet model
        x_t = self.encode_image_if_needed(x_t)
        x_t = x_t.to(self.dtype)

        self.prepare_cond(camera, text_prompt)
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        B, C, H, W = x_t.shape
        x_t = self.pipeline._pack_latents(x_t, B, C, H, W)
        # x_t_stack = torch.cat([x_t] * 2)
        controlnet_mode = [
            torch.tensor([2], device=x_t.device).expand(B),
            torch.tensor([0], device=x_t.device).expand(B),
        ]
        # print(timestep)
        timestep = torch.tensor([timestep], device=x_t.device, dtype=torch.bfloat16).expand(B)
        guidance = torch.tensor([guidance_scale], device=x_t.device).expand(B)
        # print(f"x_t: {x_t.shape}, control_images: {self.control_images[0].shape}, timestep: {timestep.shape}, guidance: {guidance.shape}")
        # print(f"controlnet_mode: {controlnet_mode[0].shape}, {controlnet_mode[1].shape}")
        # print(f"pooling: {self.cond['pooled_projections'].shape}, encoder_hidden_states: {self.cond['encoder_hidden_states'].shape}, txt_ids: {self.cond['txt_ids'].shape}, img_ids: {self.cond['img_ids'].shape}")
        # print(x_t.shape, self.control_images[0].shape, timestep, guidance)
        controlnet_block_samples, controlnet_single_block_samples = (
            self.pipeline.controlnet(
                hidden_states=x_t,
                controlnet_cond=self.control_images,
                controlnet_mode=controlnet_mode,
                conditioning_scale=conditioning_scale,
                timestep=timestep / 1000,
                guidance=guidance,
                return_dict=False,
                **self.cond
            )
        )
        noise_pred = self.pipeline.transformer(
            hidden_states=x_t,
            timestep=timestep / 1000,
            guidance=guidance,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=False,
            **self.cond
        )[0]
        # unpack
        noise_pred = self.pipeline._unpack_latents(
            noise_pred,
            H * self.pipeline.vae_scale_factor // 2,
            W * self.pipeline.vae_scale_factor // 2,
            self.pipeline.vae_scale_factor,
        )

        if return_dict:
            return {
                "noise_pred": noise_pred,
                "noise_pred_uncond": None,
                "noise_pred_text": None,
            }
        return noise_pred

    @torch.no_grad()
    def ddim_loop(
        self,
        camera,
        x_t,
        src_t,
        tgt_t,
        mode="cfg",
        guidance_scale=None,
        inv_guidance_scale=None,
        eta=0,
        num_steps=30,
        edge_preserve=False,
        clean=None,
        soft_mask=None,
        sdi_inv=False,
        try_fast=False,
        **kwargs,
    ):
        if isinstance(src_t, torch.Tensor):
            src_t = src_t.item()
        if isinstance(tgt_t, torch.Tensor):
            tgt_t = tgt_t.item()

        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        orig_dtype = x_t.dtype
        x_t = x_t.detach().to(self.dtype)

        # linearly interpolate between 1000 and 0
        # print(try_fast, edge_preserve, num_steps)
        if try_fast and edge_preserve and num_steps < 50:
            print_warning(
                "Fast sampling is disabled because edge preservation is enabled. "
                "However, num_steps is too low for DDIM. "
                "Setting num_steps to 50."
            )
            num_steps = 50
            
        raw_timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long)

        if src_t == tgt_t:
            return x_t
        elif src_t < tgt_t:
            timesteps = reversed(raw_timesteps)
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
            timesteps = raw_timesteps
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
        
        if True:
            if hasattr(self, "fast_sample"):
                # print_info("Fast sampling enabled")
                output = self.fast_sample(camera, x_t, timesteps, guidance_scale=guidance_scale, **kwargs)
                return output
        raise NotImplementedError
        if edge_preserve:
            N = len(timesteps - 1)
            H, W = x_t.shape[-2:]
            if soft_mask is not None:
                mask = soft_mask
            else:
                x = torch.linspace(0, 1, W//2)  # from 0 to 1 in 32 steps
                x = torch.cat([x, torch.flip(x, dims=[0])])  # mirror it to go back to 0 (64 steps in total)
                mask = x.repeat(H, 1).to(self.device)
            # make sure that mask is of the same size as the image
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            clean_eps = torch.randn_like(x_t)

        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):

            noise_pred_dict = self.predict(
                camera, x_t, t_curr, guidance_scale=guidance_scale, return_dict=True, **kwargs
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

            x_t = self.move_step(
                x_t, noise_pred, t_curr, t_next, renoise_eps=renoise_eps, eta=eta
            )
            if edge_preserve:
                M = (mask >= (1 - i / N)).to(x_t.dtype)
                x_t = x_t * M + self.get_noisy_sample(clean, clean_eps, t_next) * (1 - M)
                
            if sdi_inv:
                assert eta == 0, "SDI eta must be 0. It uses inversion eta to only add noise to noisy sample."
                assert t_next > t_curr, f"t_next {t_next} must be greater than t_curr {t_curr} for SDI"
                variance = self.ddim_scheduler._get_variance(t_next, t_curr) ** (0.5)
                x_t += 0.3 * torch.randn_like(x_t) * variance
                # print_info(f"sdi_inv with randn noise {variance}")

        return x_t.to(orig_dtype)