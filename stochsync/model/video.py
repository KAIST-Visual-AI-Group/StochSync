from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import read_video, write_video
from torchvision.transforms import GaussianBlur

from ..utils.extra_utils import ignore_kwargs
from ..utils.video_utils import (
    get_flow_estimator,
    forward_warping,
    backward_warping,
    get_optical_flow_raw,
)
from .. import shared_modules

from .base import BaseModel



class VideoModel(BaseModel):
    """
    Model for rendering and optimizing a set of frames.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 10000
        device: str = "cuda"
        width: int = 512
        height: int = 512
        video_path: str = "data/video.mp4"
        max_frames: int = -1

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.frames = None
        self.orig_frames = None
        self.load(self.cfg.video_path)
        self.flag = False

    @torch.no_grad()
    def load(self, path: str) -> None:
        frames, _, _ = read_video(path)
        self.orig_frames = (
            frames.permute(0, 3, 1, 2).to(torch.float32).to(self.cfg.device)
        ) / 255.0
        if self.cfg.max_frames > 0:
            self.orig_frames = self.orig_frames[: self.cfg.max_frames]
        self.orig_frames = F.interpolate(
            self.orig_frames,
            size=(self.cfg.height, self.cfg.width),
            mode="bilinear",
            align_corners=False,
        )

    @torch.no_grad()
    def save(self, path: str) -> None:
        frames = self.frames.permute(0, 2, 3, 1).cpu() * 255.0  # (N, C, H, W) -> (N, H, W, C)
        frames = frames.to(torch.uint8).numpy()
        write_video(path, frames, 30)

    def prepare_optimization(self) -> None:
        self.frames = self.orig_frames.clone().to(self.cfg.device)
        self.estimator = get_flow_estimator(self.cfg.device)
        # self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)

    def render(self, camera) -> torch.Tensor:
        frames = F.interpolate(
            self.frames,
            size=(camera["height"], camera["width"]),
            mode="bilinear",
            align_corners=False,
        )

        rendered = []
        for idx in camera["frame_idx"]:
            rendered.append(frames[idx])
        rendered = torch.stack(rendered)

        return {
            "image": rendered,
            "alpha": torch.ones(
                rendered.shape[0],
                1,
                rendered.shape[2],
                rendered.shape[3],
                device=self.cfg.device,
            ),
        }
    
    @torch.no_grad()
    def render_eval(self) -> torch.Tensor:
        pass

    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        num_frames = self.frames.shape[0]
        vnum = max(1, int(num_frames**0.5))
        hnum = (num_frames - 1) // vnum + 1
        image = torch.zeros(
            1, 3, self.cfg.height * vnum, self.cfg.width * hnum, device=self.cfg.device
        )
        for i in range(num_frames):
            vidx = i // hnum
            hidx = i % hnum
            image[
                :,
                :,
                vidx * self.cfg.height : (vidx + 1) * self.cfg.height,
                hidx * self.cfg.width : (hidx + 1) * self.cfg.width,
            ] = self.frames[i]
        # downscale to 0.5, 0.5
        image = F.interpolate(
            image, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        return image

    def optimize(self, step: int) -> None:
        raise NotImplementedError

    def closed_form_optimize(self, step, camera, target):
        target = shared_modules.prior.decode_latent_if_needed(target)

        num_frames = self.frames.shape[0]
        indices = camera["frame_idx"]

        # if self.flag:
        #     # invert indices, next_indices, target, and frames
        #     indices = [num_frames - 1 - i for i in indices[::-1]]
        #     target = target.flip(0)
        #     self.frames = self.frames.flip(0)
        #     self.orig_frames = self.orig_frames.flip(0)

        next_indices = indices[1:] + [num_frames]

        warping_pairs = []
        cnt = 0
        for idx, next_idx in zip(indices, next_indices):
            while cnt <= (idx + next_idx) // 2:
                warping_pairs.append((idx, cnt))
                cnt += 1
        
        for idx, tgt in zip(indices, target):
            self.frames[idx] = tgt
        
        for idx, next_idx in warping_pairs:
            # flow = get_optical_flow_raw(
            #     self.orig_frames[next_idx][None],
            #     self.orig_frames[idx][None],
            #     model=self.estimator,
            # )
            # warped, mask = backward_warping(self.frames[idx:idx+1], flow, return_mask=True)
            # mask = mask.expand_as(warped)
            # self.frames[next_idx : next_idx + 1] = warped * mask + self.frames[next_idx : next_idx + 1] * (1 - mask)

            flow = get_optical_flow_raw(
                self.orig_frames[idx][None],
                self.orig_frames[next_idx][None],
                model=self.estimator,
            )
            warped, mask = forward_warping(self.frames[idx:idx+1], flow, return_mask=True)
            mask = mask.float()
            # gaussian kernel using PyTorch

            mask = GaussianBlur(9, 3)(mask) * mask


            mask = mask.expand_as(warped)
            tmp = warped * mask + self.frames[next_idx : next_idx + 1] * (1 - mask)
            # self.frames[next_idx : next_idx + 1] = tmp
            self.frames[next_idx : next_idx + 1] += tmp
            self.frames[next_idx : next_idx + 1] /= 2
        

        # if self.flag:
        #     indices = [num_frames - 1 - i for i in indices[::-1]]
        #     next_indices = [num_frames - 1 - i for i in indices[:-1:-1]] + [next_indices[-1]]
        #     target = target.flip(0)
        #     self.frames = self.frames.flip(0)
        #     self.orig_frames = self.orig_frames.flip(0)
        
        # self.flag = not self.flag

    def regularize(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.cfg.device)
