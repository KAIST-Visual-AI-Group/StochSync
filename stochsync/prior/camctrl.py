from typing import Dict, Literal, Tuple, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch
import diffusers
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionGLIGENPipeline,
    DDIMScheduler,
)
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from third_party.mvdream.pipeline_mvdream import MVDreamPipeline

from ..utils.matrix_utils import rodrigues
from ..utils.camera_utils import convert_camera_convention
from ..utils.extra_utils import (
    attach_direction_prompt,
    ignore_kwargs,
    attach_elevation_prompt,
    weak_lru,
)
from ..utils.print_utils import print_info, print_warning, print_error

from .base import Prior, NEGATIVE_PROMPT

# CameraCtrl setup
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from packaging import version as pver
from einops import rearrange
from safetensors import safe_open

from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_vae_checkpoint,
    convert_ldm_clip_checkpoint,
)

from .cameractrl.utils.util import save_videos_grid
from .cameractrl.models.unet import UNet3DConditionModelPoseCond
from .cameractrl.models.pose_adaptor import CameraPoseEncoder
from .cameractrl.pipelines.pipeline_animation import CameraCtrlPipeline
from .cameractrl.utils.convert_from_ckpt import convert_ldm_unet_checkpoint
from .cameractrl.data.dataset import Camera


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def get_relative_pose(c2ws):
    abs_w2cs = np.linalg.inv(c2ws)
    abs_c2ws = c2ws
    cam_to_origin = 0
    target_cam_c2w = np.array(
        [[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [
        target_cam_c2w,
    ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def load_personalized_base_model(pipeline, personalized_base_model):
    print(f"Load civitai base model from {personalized_base_model}")
    if personalized_base_model.endswith(".safetensors"):
        dreambooth_state_dict = {}
        with safe_open(personalized_base_model, framework="pt", device="cpu") as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)
    elif personalized_base_model.endswith(".ckpt"):
        dreambooth_state_dict = torch.load(personalized_base_model, map_location="cpu")

    # 1. vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(
        dreambooth_state_dict, pipeline.vae.config
    )
    pipeline.vae.load_state_dict(converted_vae_checkpoint)
    # 2. unet
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        dreambooth_state_dict, pipeline.unet.config
    )
    _, unetu = pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
    assert len(unetu) == 0
    # 3. text_model
    pipeline.text_encoder = convert_ldm_clip_checkpoint(
        dreambooth_state_dict, text_encoder=pipeline.text_encoder
    )
    del dreambooth_state_dict
    return pipeline


def get_pipeline(
    ori_model_path,
    unet_subfolder,
    image_lora_rank,
    image_lora_ckpt,
    unet_additional_kwargs,
    unet_mm_ckpt,
    pose_encoder_kwargs,
    attention_processor_kwargs,
    noise_scheduler_kwargs,
    pose_adaptor_ckpt,
    personalized_base_model,
    gpu_id,
):
    vae = AutoencoderKL.from_pretrained(ori_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(ori_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        ori_model_path, subfolder="text_encoder"
    )
    unet = UNet3DConditionModelPoseCond.from_pretrained_2d(
        ori_model_path,
        subfolder=unet_subfolder,
        unet_additional_kwargs=unet_additional_kwargs,
    )
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)
    print(f"Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=image_lora_ckpt is not None,
        add_motion_lora=False,
        lora_kwargs={"lora_rank": image_lora_rank, "lora_scale": 1.0},
        motion_lora_kwargs={"lora_rank": -1, "lora_scale": 1.0},
        **attention_processor_kwargs,
    )

    if image_lora_ckpt is not None:
        print(f"Loading the lora checkpoint from {image_lora_ckpt}")
        lora_checkpoints = torch.load(image_lora_ckpt, map_location=unet.device)
        if "lora_state_dict" in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints["lora_state_dict"]
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        print(f"Loading done")

    if unet_mm_ckpt is not None:
        print(f"Loading the motion module checkpoint from {unet_mm_ckpt}")
        mm_checkpoints = torch.load(unet_mm_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        print("Loading done")

    print(f"Loading pose adaptor")
    pose_adaptor_checkpoint = torch.load(pose_adaptor_ckpt, map_location="cpu")
    pose_encoder_state_dict = pose_adaptor_checkpoint["pose_encoder_state_dict"]
    pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(
        pose_encoder_state_dict
    )
    assert len(pose_encoder_u) == 0 and len(pose_encoder_m) == 0
    attention_processor_state_dict = pose_adaptor_checkpoint[
        "attention_processor_state_dict"
    ]
    _, attn_proc_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attn_proc_u) == 0
    print(f"Loading done")

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae.to(gpu_id)
    text_encoder.to(gpu_id)
    unet.to(gpu_id)
    pose_encoder.to(gpu_id)
    pipe = CameraCtrlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        pose_encoder=pose_encoder,
    )
    if personalized_base_model is not None:
        load_personalized_base_model(
            pipeline=pipe, personalized_base_model=personalized_base_model
        )
    pipe.enable_vae_slicing()
    pipe = pipe.to(gpu_id)

    return pipe


class CameraCtrlPrior(Prior):
    @ignore_kwargs
    @dataclass
    class Config:
        model_name: str = "runwayml/stable-diffusion-v1-5"
        text_prompt: str = (
            "a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
        )
        negative_prompt: str = NEGATIVE_PROMPT
        width: int = 384
        height: int = 256
        guidance_scale: int = 7.5
        mixed_precision: bool = False
        root_dir: str = "./results/default"

        single_model_length: int = 16
        md_num: int = 3

    def __init__(self, cfg):
        super().__init__()
        self.cfg = self.Config(**cfg)

        file_dir = os.path.dirname(os.path.abspath(__file__))

        model_configs = OmegaConf.load(os.path.join(file_dir, "cameractrl/config/adv3_256_384_cameractrl_relora.yaml"))
        unet_additional_kwargs = model_configs[
            'unet_additional_kwargs'] if 'unet_additional_kwargs' in model_configs else None
        noise_scheduler_kwargs = model_configs['noise_scheduler_kwargs']
        pose_encoder_kwargs = model_configs['pose_encoder_kwargs']
        attention_processor_kwargs = model_configs['attention_processor_kwargs']

        self.ddim_scheduler = None
        self.fast_scheduler = None
        self.pipeline = get_pipeline(
            os.path.join(file_dir, "cameractrl/sd15"),
            "lora",
            2,
            None,
            unet_additional_kwargs,
            os.path.join(file_dir, "cameractrl/ckpts/v3_sd15_mm.ckpt"),
            pose_encoder_kwargs,
            attention_processor_kwargs,
            noise_scheduler_kwargs,
            os.path.join(file_dir, "cameractrl/ckpts/CameraCtrl.ckpt"),
            None,
            "cuda:0",
        )
        self.ddim_scheduler = self.pipeline.scheduler
        self.fast_scheduler = self.pipeline.scheduler
        
        self.ddim_scheduler.set_timesteps(30)
        self.pipeline.unet.requires_grad_(False)
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.pose_encoder.requires_grad_(False)
        print_info(self.cfg.text_prompt)

    @property
    def rgb_res(self):
        return 1, 3, 256, 384

    @property
    def latent_res(self):
        return 1, 4, 32, 48

    def prepare_cond(self, camera, text_prompt=None, negative_prompt=None):
        text_prompt = text_prompt if text_prompt is not None else self.cfg.text_prompt
        negative_prompt = (
            negative_prompt if negative_prompt is not None else self.cfg.negative_prompt
        )

        text_prompts = [text_prompt]

        neg_embeds, pos_embeds = [], []
        for prompt in text_prompts:
            text_embeddings = self.encode_text(
                prompt, negative_prompt=negative_prompt
            )  # neg, pos
            neg, pos = text_embeddings.chunk(2)
            neg_embeds.append(neg)
            pos_embeds.append(pos)

        text_embeddings = torch.cat(neg_embeds + pos_embeds, dim=0)

        Ks = camera["K"][:16]
        azims = camera["azimuth"][:16]
        # make the first element zero
        azims = [azim - azims[0] for azim in azims]

        # make c2ws(y-axis rotation matrices)
        rots = []
        for azim in azims:
            rot = rodrigues(torch.tensor([0, 1, 0]).float(), azim * np.pi / 180).to(Ks.device)
            rots.append(rot)
        rots = torch.stack(rots, dim=0)
        c2ws = torch.eye(4).unsqueeze(0).repeat(len(rots), 1, 1).to(Ks.device)
        c2ws[:, :3, :3] = rots
        # print(c2ws)

        # B 3 3 -> 1 B 4
        Ks = torch.stack([Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]], dim=-1).unsqueeze(0)

        # B 4 4 -> 1 B 4 4
        c2ws = c2ws.unsqueeze(0)

        _, _, H, W = self.rgb_res

        plucker_embedding = (
            ray_condition(Ks.cpu(), c2ws.cpu(), H, W, device="cpu")[0]
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # V, 6, H, W
        plucker_embedding = plucker_embedding[None].to("cuda")  # B V 6 H W
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")

        pose_embedding_features = self.pipeline.pose_encoder(
            plucker_embedding
        )  # bf, c, h, w

        bs = plucker_embedding.shape[0]
        pose_embedding_features = [
            rearrange(x, "(b f) c h w -> b c f h w", b=bs)
            for x in pose_embedding_features
        ]
        pose_embedding_features = [
            torch.cat([x, x], dim=0) for x in pose_embedding_features
        ]  # [2b c f h w]

        self.cond = {
            "encoder_hidden_states": text_embeddings,
            "pose_embedding_features": pose_embedding_features,
        }
        return self.cond

    def sample(self, camera, text_prompt=None):
        if text_prompt is None:
            text_prompt = self.cfg.text_prompt

        self.prepare_cond(camera)
        with torch.no_grad():
            images = self.pipeline(
                [text_prompt], negative_prompt=[self.cfg.negative_prompt]
            ).images
        return images

    def fast_sample(
        self,
        camera,
        x_t,
        timesteps,
        guidance_scale=None,
        text_prompt=None,
        negative_prompt=None,
    ):
        # self.fast_scheduler.set_timesteps(timesteps)
        self.fast_scheduler.num_inference_steps = len(timesteps)
        self.fast_scheduler.timesteps = timesteps
        for t in timesteps:
            noise_pred = self.predict(
                camera, x_t, t, guidance_scale, text_prompt, negative_prompt
            )
            x_t = self.fast_scheduler.step(noise_pred, t, x_t, return_dict=False)[0]

        return x_t

    def predict(
        self,
        camera,
        x_t,
        timestep,
        guidance_scale=None,
        return_dict=False,
        text_prompt=None,
        negative_prompt=None,
    ):
        # Predict the noise using the UNet model
        if x_t.shape[1] == 3:
            x_t = self.encode_image(x_t)

        # B C H W -> 1 C B H W
        x_t = x_t.unsqueeze(0).permute(0, 2, 1, 3, 4)

        self.prepare_cond(camera, text_prompt, negative_prompt)
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        )

        noise_pred_full_text = torch.zeros_like(x_t).to(x_t.device)
        noise_pred_full_uncond = torch.zeros_like(x_t).to(x_t.device)
        mask_full = torch.zeros_like(x_t).to(x_t.device)
        noise_preds_text = []
        noise_preds_uncond = []

        md_num = self.cfg.md_num
        single_model_length = self.cfg.single_model_length
        batch_size = x_t.shape[2]
        assert batch_size % md_num == 0, "Batch size must be divisible by md_num"
        overlap = single_model_length - batch_size // md_num

        for multidiff_step in range(md_num):
            i = multidiff_step * (single_model_length - overlap)
            end_idx = i + single_model_length
            if end_idx > x_t.shape[2]:
                latent_partial = torch.cat(
                    (x_t[:, :, i:], x_t[:, :, :end_idx - x_t.shape[2]]), dim=2
                ).contiguous()
                mask_full[:, :, i:] += 1
                mask_full[:, :, :end_idx - x_t.shape[2]] += 1
            else:
                latent_partial = x_t[:, :, i:end_idx].contiguous()
                mask_full[:, :, i:end_idx] += 1

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latent_partial] * 2)  # [2b c f h w]

            # predict the noise residual
            noise_pred = self.pipeline.unet(latent_model_input, timestep, **self.cond).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_preds_text.append(noise_pred_text)
            noise_preds_uncond.append(noise_pred_uncond)
        
        for pred_idx, (noise_pred_text, noise_pred_uncond) in enumerate(
            zip(noise_preds_text, noise_preds_uncond)
        ):
            i = pred_idx * (single_model_length - overlap)
            end_idx = i + single_model_length
            if end_idx > x_t.shape[2]:
                _head, _tail = x_t.shape[2] - i, end_idx - x_t.shape[2]
                noise_pred_full_text[:, :, i:] += (
                    noise_pred_text[:, :, :_head] / mask_full[:, :, i:]
                )
                noise_pred_full_text[:, :, :_tail] += (
                    noise_pred_text[:, :, _head:] / mask_full[:, :, :_tail]
                )
                noise_pred_full_uncond[:, :, i:] += (
                    noise_pred_uncond[:, :, :_head] / mask_full[:, :, i:]
                )
                noise_pred_full_uncond[:, :, :_tail] += (
                    noise_pred_uncond[:, :, _head:] / mask_full[:, :, :_tail]
                )
            else:
                _slice = slice(i, end_idx)
                noise_pred_full_text[:, :, _slice] += (
                    noise_pred_text / mask_full[:, :, _slice]
                )
                noise_pred_full_uncond[:, :, _slice] += (
                    noise_pred_uncond / mask_full[:, :, _slice]
                )

        noise_pred_full = noise_pred_full_uncond + guidance_scale * (
            noise_pred_full_text - noise_pred_full_uncond
        )

        # 1 C B H W -> B C H W
        noise_pred_full = noise_pred_full.permute(0, 2, 1, 3, 4).squeeze(0)
        noise_pred_full_uncond = noise_pred_full_uncond.permute(0, 2, 1, 3, 4).squeeze(0)
        noise_pred_full_text = noise_pred_full_text.permute(0, 2, 1, 3, 4).squeeze(0)

        if return_dict:
            return {
                "noise_pred": noise_pred_full,
                "noise_pred_uncond": noise_pred_full_uncond,
                "noise_pred_text": noise_pred_full_text,
            }
        return noise_pred_full

    @weak_lru(maxsize=10)
    def encode_text(self, prompt, negative_prompt=None):
        """
        Encode a text prompt into a feature vector.
        """
        assert self.pipeline is not None, "Pipeline not initialized"
        text_embeddings = self.pipeline._encode_prompt(
            prompt, "cuda", 1, True, negative_prompt=negative_prompt
        )
        # uncond, cond
        # text_embeddings = torch.stack([text_embeddings[1], text_embeddings[0]])
        return text_embeddings