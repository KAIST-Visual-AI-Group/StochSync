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
from pytorch3d.renderer import FoVPerspectiveCameras, TexturesUV

from .nvdiff_render.mesh import *
from .nvdiff_render.render import *
from .nvdiff_render.texture import *
from .nvdiff_render.material import *
from .nvdiff_render.obj import *

from .base import BaseModel

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
import shared_modules
from model.mesh_utils.mesh_renderer import Renderer
from .dc_pbr import skip

from k_utils.image_utils import save_tensor, pil_to_torch
from k_utils.print_utils import print_info

glctx = dr.RasterizeCudaContext()

class MeshModel(BaseModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        device: str = "cuda"
        mesh_path: str = ""
        texture_size: int = 1024
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
        self.renderer = None
        self.optimizer = None

        self.texture = None
        self.mesh = None
        self.load(self.cfg.mesh_path)

    def load(self, path: str) -> None:
        obj_f_uv, obj_v_uv, obj_f, obj_v = load_obj_uv(obj_path=self.cfg.mesh_path, device=self.cfg.device)

        # initialize template mesh
        self.mesh = Mesh(obj_v, obj_f, v_tex=obj_v_uv, t_tex_idx=obj_f_uv)
        self.mesh = unit_size(self.mesh)
        self.mesh = auto_normals(self.mesh)
        self.mesh = compute_tangents(self.mesh)

    def save(self, path: str) -> None:
        pass

    def prepare_optimization(self) -> None:
        self.texture = torch.full((1, self.cfg.texture_size, self.cfg.texture_size, 3), 0.5, device=self.cfg.device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.texture], self.cfg.learning_rate, weight_decay=self.cfg.decay)

    def render(self, camera) -> torch.Tensor:

        kd_min = [0.0, 0.0, 0.0, 0.0]  # Limits for kd
        kd_max = [1.0, 1.0, 1.0, 1.0]
        kd_min, kd_max = torch.tensor(kd_min, dtype=torch.float32, device='cuda'), torch.tensor(kd_max, dtype=torch.float32, device='cuda')

        pred_material = Material({
            # 'bsdf': 'pbr',
            'bsdf': 'kd', # Jaihoon
            'kd': Texture2D(self.texture, min_max=[kd_min, kd_max]),
        })
        #pred_material['kd'].clamp_()

        self.mesh.material = pred_material

        buffers = render_mesh(glctx, self.mesh, camera['mvp'], camera['campos'], None, camera['resolution'],
                                spp=camera['spp'], msaa=True, background=None, bsdf='kd')
        pred_obj_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        pred_obj_ws = buffers['shaded'][..., 3].unsqueeze(1)  # [B, 1, H, W]
        obj_image = pred_obj_rgb


        return {
            "image": obj_image,
            #"alpha": torch.ones_like(pred_obj_ws),
            "alpha": pred_obj_ws.detach(),
        }

    def optimize(self, step: int) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()

from pytorch3d.io import load_obj, save_obj
def load_obj_uv(obj_path, device):
    vert, face, aux = load_obj(obj_path, device=device)
    vt = aux.verts_uvs
    ft = face.textures_idx
    vt = torch.cat((vt[:, [0]], 1.0 - vt[:, [1]]), dim=1)
    return ft, vt, face.verts_idx, vert

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
        self.renderer = None
        self.optimizer = None

        self.network_input = None
        self.net = None
        self.mesh = None
        self.load(self.cfg.mesh_path)

    def load(self, path: str) -> None:
        obj_f_uv, obj_v_uv, obj_f, obj_v = load_obj_uv(obj_path=self.cfg.mesh_path, device=self.cfg.device)

        # initialize template mesh
        self.mesh = Mesh(obj_v, obj_f, v_tex=obj_v_uv, t_tex_idx=obj_f_uv)
        self.mesh = unit_size(self.mesh)
        self.mesh = auto_normals(self.mesh)
        self.mesh = compute_tangents(self.mesh)

    def save(self, path: str) -> None:
        pass

    def prepare_optimization(self) -> None:
        input_uv_ = torch.randn((3, self.cfg.texture_size, self.cfg.texture_size), device=self.cfg.device)
        input_uv = (
            input_uv_ - torch.mean(input_uv_, dim=(1, 2)).reshape(-1, 1, 1)
        ) / torch.std(input_uv_, dim=(1, 2)).reshape(-1, 1, 1)
        self.network_input = copy.deepcopy(input_uv.unsqueeze(0))

        self.net, self.optimizer, activate_scheduler, self.lr_scheduler = get_model(
            self.cfg
        )
        print("For debugging")
        self.texture = Image.open("bird.png")
        # resize
        self.texture = self.texture.resize((self.cfg.texture_size, self.cfg.texture_size))
        self.texture = pil_to_torch(self.texture).to(self.cfg.device).permute(0, 2, 3, 1)

    def render(self, camera) -> torch.Tensor:

        kd_min = [0.0, 0.0, 0.0, 0.0]  # Limits for kd
        kd_max = [1.0, 1.0, 1.0, 1.0]
        #ks_min = [0.0, 0.08, 0.0]  # Limits for ks
        #ks_max = [1.0, 1.0, 1.0]
        #nrm_min = [-0.1, -0.1, 0.0]  # Limits for normal map
        #nrm_max = [0.1, 0.1, 1.0]
        kd_min, kd_max = torch.tensor(kd_min, dtype=torch.float32, device='cuda'), torch.tensor(kd_max, dtype=torch.float32, device='cuda')
        #ks_min, ks_max = torch.tensor(ks_min, dtype=torch.float32, device='cuda'), torch.tensor(ks_max, dtype=torch.float32, device='cuda')
        #nrm_min, nrm_max = torch.tensor(nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(nrm_max, dtype=torch.float32, device='cuda')
        #nrm_t = get_template_normal(h=self.cfg.texture_size, w=self.cfg.texture_size)

        net_output = self.net(self.network_input)  # [B, 9, H, W]
        pred_tex = net_output.permute(0, 2, 3, 1)
        pred_kd = pred_tex[..., :-6]
        #pred_ks = pred_tex[..., -6:-3]
        #pred_n = F.normalize((pred_tex[..., -3:] * 2.0 - 1.0) + nrm_t, dim=-1)

        pred_material = Material({
            # 'bsdf': 'pbr',
            'bsdf': 'kd', # Jaihoon
            #'kd': Texture2D(self.texture + 0*pred_kd, min_max=[kd_min, kd_max]),
            'kd': Texture2D(pred_kd, min_max=[kd_min, kd_max]),
            #'ks': Texture2D(pred_ks, min_max=[ks_min, ks_max]),
            #'normal': Texture2D(pred_n, min_max=[nrm_min, nrm_max])
        })
        pred_material['kd'].clamp_()
        #pred_material['ks'].clamp_()
        #pred_material['normal'].clamp_()

        # Texture settings
        # net_output = self.net(self.network_input)  # [B, 3, H, W]
        # net_output = net_output.permute(0, 2, 3, 1).contiguous()
        # print(net_output.shape)


        # pred_material = Material({
        #     'bsdf': 'kd',
        #     'kd': Texture2D(net_output, min_max=[torch.Tensor([0.0, 0.0, 0.0, 0.0]).to(net_output.device), torch.Tensor([1.0, 1.0, 1.0, 1.0]).to(net_output.device)])
        # })

        #with torch.no_grad():
        #    mesh = copy.deepcopy(self.mesh)
        self.mesh.material = pred_material

        buffers = render_mesh(glctx, self.mesh, camera['mvp'], camera['campos'], None, camera['resolution'],
                                spp=camera['spp'], msaa=True, background=None, bsdf='kd')
        pred_obj_rgb = buffers['shaded'][..., 0:3].permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        pred_obj_ws = buffers['shaded'][..., 3].unsqueeze(1)  # [B, 1, H, W]
        obj_image = pred_obj_rgb


        return {
            "image": obj_image,
            "alpha": pred_obj_ws.detach(),
        }

    def optimize(self, step: int) -> None:
        #print(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()


def get_model(cfg):
    # MLP Settings
    net = skip(
        3,
        9,
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
    activate_scheduler = (
        cfg.lr_decay < 1 and cfg.decay_step > 0 and not cfg.lr_plateau
    )
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=cfg.decay_step, gamma=cfg.lr_decay
        )
    
    return net, optim, activate_scheduler, lr_scheduler

def get_template_normal(h=512, w=512):
    return torch.cat([torch.zeros((h, w, 1), device="cuda"), torch.zeros((h, w, 1), device="cuda"),
                      torch.ones((h, w, 1), device="cuda")], dim=-1)[None, ...]