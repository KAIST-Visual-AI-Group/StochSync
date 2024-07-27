import os
import sys
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
import trimesh

from .nvdiff_render.mesh import *
from .nvdiff_render.render import *
from .nvdiff_render.texture import *
from .nvdiff_render.material import *
from .nvdiff_render.obj import *

from .base import BaseModel

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
import shared_modules

# from model.mesh_utils.mesh_renderer import Renderer
from .dc_pbr import skip

from k_utils.image_utils import save_tensor, pil_to_torch
from k_utils.print_utils import print_info, print_warning


class MeshModel(BaseModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        device: str = "cuda"
        mesh_path: str = "bird.obj"
        texture_size: int = 1024
        mesh_scale: float = 1.0
        sampling_mode: str = "nearest"
        initialization: str = "random"  # random, zero
        channels: int = 3

        learning_rate: float = 0.0005
        decay: float = 0
        lr_decay: float = 0.9
        decay_step: int = 100
        lr_plateau: bool = False

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.optimizer = None

        self.glctx = dr.RasterizeCudaContext()
        self.texture = None
        self.mesh = None
        self.load(self.cfg.mesh_path)

    def load(self, path: str) -> None:
        f_idx, v_pos, v_uv = load_obj_uv(obj_path=path, device=self.cfg.device)
        self.mesh = Mesh(v_pos, f_idx, v_tex=v_uv, t_tex_idx=f_idx)
        self.mesh = unit_size(self.mesh)
        self.mesh = auto_normals(self.mesh)
        self.mesh = compute_tangents(self.mesh)

    def save(self, path: str) -> None:
        pass

    def prepare_optimization(self) -> None:
        shape = (1, self.cfg.channels, self.cfg.texture_size, self.cfg.texture_size)
        if self.cfg.initialization == "random":
            self.texture = torch.randn(
                shape, device=self.cfg.device, requires_grad=True
            )
        elif self.cfg.initialization == "zero":
            self.texture = torch.zeros(
                shape, device=self.cfg.device, requires_grad=True
            )
        elif self.cfg.initialization == "gray":
            self.texture = torch.full(
                shape, 0.5, device=self.cfg.device, requires_grad=True
            )
        else:
            raise ValueError(f"Invalid initialization: {self.cfg.initialization}")

        # print("For debugging")
        # self.texture = Image.open("bird.png")
        # self.texture = self.texture.resize(
        #     (self.cfg.texture_size, self.cfg.texture_size)
        # )
        # self.texture = (
        #     pil_to_torch(self.texture).to(self.cfg.device).permute(0, 2, 3, 1)
        # )

        self.optimizer = torch.optim.Adam(
            [self.texture], self.cfg.learning_rate, weight_decay=self.cfg.decay
        )

    def render(self, camera) -> torch.Tensor:
        c2ws, Ks, width, height, fov = (
            camera["c2w"],
            camera["K"],
            camera["width"],
            camera["height"],
            camera["fov"],
        )
        proj_mtx = util.perspective(fov * np.pi / 180, width / height, 0.1, 1000.0).to(
            self.cfg.device
        )
        mv = c2ws.inverse()
        mvp = proj_mtx @ mv
        campos = c2ws[:, :3, 3]

        texture = self.texture.permute(0, 2, 3, 1)  # .clamp(0, 1)
        pred_material = Material(
            {
                "bsdf": "kd",
                "kd": Texture2D(texture),
            }
        )

        self.mesh.material = pred_material

        render_pkg = render_mesh(
            self.glctx,
            self.mesh,
            mvp,  # B 4 4
            campos,  # B 3
            None,
            [height, width],
            msaa=True,
            background=None,
            channels=self.cfg.channels,
        )
        image = (
            render_pkg["shaded"][..., :-1].permute(0, 3, 1, 2).contiguous()
        )  # [B, 3, H, W]
        alpha = render_pkg["shaded"][..., -1].unsqueeze(1)  # [B, 1, H, W]

        return {
            "image": image,
            "alpha": alpha.detach(),
        }

    def optimize(self, step: int) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()


def load_obj_uv(obj_path, device):
    # Load the obj file using trimesh
    mesh = trimesh.load(obj_path, process=False)

    # Extract vertex positions
    v_coords = torch.tensor(mesh.vertices, device=device).float()

    # Extract face indices
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)

    # Extract texture UV coordinates
    uv_coords = torch.tensor(mesh.visual.uv, device=device).float()
    uv_coords = torch.cat((uv_coords[:, [0]], 1.0 - uv_coords[:, [1]]), dim=1)

    return faces, v_coords, uv_coords


class PaintitMeshModel(BaseModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        device: str = "cuda"
        mesh_path: str = ""
        texture_size: int = 512
        mesh_scale: float = 1.0
        sampling_mode: str = "nearest"
        initialization: str = "gray"  # random, zero
        channels: int = 3

        learning_rate: float = 0.0005
        decay: float = 0
        lr_decay: float = 0.9
        decay_step: int = 100
        lr_plateau: bool = False

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.optimizer = None

        self.glctx = dr.RasterizeCudaContext()
        self.network_input = None
        self.net = None
        self.mesh = None
        self.load(self.cfg.mesh_path)

    def load(self, path: str) -> None:
        f_idx, v_pos, v_uv = load_obj_uv(obj_path=path, device=self.cfg.device)
        self.mesh = Mesh(v_pos, f_idx, v_tex=v_uv, t_tex_idx=f_idx)
        self.mesh = unit_size(self.mesh)
        self.mesh = auto_normals(self.mesh)
        self.mesh = compute_tangents(self.mesh)

    def save(self, path: str) -> None:
        pass

    def prepare_optimization(self) -> None:
        input_uv_ = torch.randn(
            (3, self.cfg.texture_size, self.cfg.texture_size), device=self.cfg.device
        )
        input_uv = (
            input_uv_ - torch.mean(input_uv_, dim=(1, 2)).reshape(-1, 1, 1)
        ) / torch.std(input_uv_, dim=(1, 2)).reshape(-1, 1, 1)
        self.network_input = copy.deepcopy(input_uv.unsqueeze(0))

        self.net, self.optimizer, activate_scheduler, self.lr_scheduler = get_model(
            self.cfg
        )

    def render(self, camera):
        c2ws, Ks, width, height, fov = (
            camera["c2w"],
            camera["K"],
            camera["width"],
            camera["height"],
            camera["fov"],
        )
        proj_mtx = util.perspective(fov * np.pi / 180, width / height, 0.1, 1000.0).to(
            self.cfg.device
        )
        mv = c2ws.inverse()
        mvp = proj_mtx @ mv
        campos = c2ws[:, :3, 3]

        net_output = self.net(self.network_input)  # [B, 3, H, W]
        texture = net_output.permute(0, 2, 3, 1).clamp(0, 1)
        pred_material = Material(
            {
                "bsdf": "kd",
                "kd": Texture2D(texture),
            }
        )
        self.mesh.material = pred_material

        render_pkg = render_mesh(
            self.glctx,
            self.mesh,
            mvp,  # B 4 4
            campos,  # B 3
            None,
            [height, width],
            msaa=True,
            background=None,
            channels=self.cfg.channels,
        )
        image = (
            render_pkg["shaded"][..., :-1].permute(0, 3, 1, 2).contiguous()
        )  # [B, 3, H, W]
        alpha = render_pkg["shaded"][..., -1].unsqueeze(1)  # [B, 1, H, W]

        return {
            "image": image,
            "alpha": alpha.detach(),
        }

    def optimize(self, step: int) -> None:
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()


def get_model(cfg):
    # MLP Settings
    net = skip(
        3,
        3,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5,
        filter_size_up=3,
        filter_size_down=3,
        upsample_mode="nearest",
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad="reflection",
        act_fun="LeakyReLU",
    ).type(torch.cuda.FloatTensor)

    params = list(net.parameters())

    optim = torch.optim.Adam(params, cfg.learning_rate, weight_decay=cfg.decay)
    activate_scheduler = cfg.lr_decay < 1 and cfg.decay_step > 0 and not cfg.lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=cfg.decay_step, gamma=cfg.lr_decay
        )

    return net, optim, activate_scheduler, lr_scheduler
