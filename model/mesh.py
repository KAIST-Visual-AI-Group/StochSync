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

from nvdiffrast.torch import rasterize, interpolate  # low-level rasterization functions
from .nvdiff_render.mesh import *
from .nvdiff_render.render import *
from .nvdiff_render.texture import *
from .nvdiff_render.material import *
from .nvdiff_render.obj import *

from .base import BaseModel

from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
import shared_modules as sm

# from model.mesh_utils.mesh_renderer import Renderer
from .dc_pbr import skip

from k_utils.image_utils import save_tensor, pil_to_torch
from k_utils.print_utils import print_info, print_warning

from math import ceil
from torchvision.transforms import GaussianBlur
from utils.panorama_utils import remap

from utils.camera_utils import index_camera

from torch.nn import CircularPad2d


def smooth_mask(mask, kernel_coeff=0.006, threshold=0.9):
    # mask: (B, H, W)
    height, width = mask.shape[-2:]
    sigma = kernel_coeff * min(width, height)
    kernel_size = 2 * ceil(3 * sigma) + 1

    pad_size = kernel_size // 2
    padded_mask = CircularPad2d(pad_size)(mask)
    padded_blur = GaussianBlur(kernel_size, sigma)(padded_mask.unsqueeze(1).float()).squeeze(1)
    blur = padded_blur[:, pad_size:-pad_size, pad_size:-pad_size]

    return mask & (blur > threshold)

def unproject(camera, target, coordmap, return_mask=False):
    c2ws, Ks, width, height, fov = (
        camera["c2w"],
        camera["K"],
        camera["width"],
        camera["height"],
        camera["fov"],
    )
    proj_mtx = util.perspective(fov * np.pi / 180, width / height, 0.1, 1000.0).to(
        c2ws.device
    )
    mv = c2ws.inverse()
    mvp = proj_mtx @ mv

    gb_pos_clip = torch.einsum("bhwp,bpq->bhwq", coordmap, mvp.permute(0, 2, 1))
    gb_pos_clip = gb_pos_clip[..., :3] / gb_pos_clip[..., 3:4]

    gp_pos_xy = (0.5 * gb_pos_clip[..., :2] + 0.5) * 512
    gp_pos_z = gb_pos_clip[..., 2]
    front_z = sm.model.render(camera, bsdf="depth")["image"]
    gp_front_z = remap(front_z[:, :1], gp_pos_xy.squeeze()).squeeze(1)
    mask = torch.isclose(gp_front_z, gp_pos_z, atol=0.002)

    cos = sm.model.render(camera, bsdf="cos")['image'][:, :1]
    # print(cos.shape)
    cos_view_mask = (cos > 0.5).float()
    cos_mask = remap(cos_view_mask, gp_pos_xy.squeeze(), mode="nearest")
    # target = target * cos_mask

    unproj_texture = remap(target, gp_pos_xy.squeeze()) * mask * cos_mask

    # save_tensor(unproj_texture, "unproj_texture.png")
    # save_tensor(mask.float(), "mask.png", is_grayscale=True)
    # save_tensor(cos_mask.squeeze().float(), "cos_mask.png", is_grayscale=True)
    # exit()

    if return_mask:
        return unproj_texture, mask * cos_mask
    return unproj_texture


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

        dist_range: Tuple[float, float] = (2.0, 2.0)

        use_selection: bool = False

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.optimizer = None

        self.glctx = dr.RasterizeCudaContext()
        self.texture = None
        self.mesh = None
        self.load(self.cfg.mesh_path)

    @torch.no_grad()
    def load(self, path: str) -> None:
        f_idx, v_pos, v_uv = load_obj_uv(obj_path=path, device=self.cfg.device)
        self.mesh = Mesh(v_pos, f_idx, v_tex=v_uv, t_tex_idx=f_idx)
        self.mesh = unit_size(self.mesh)
        self.mesh = auto_normals(self.mesh)
        self.mesh = compute_tangents(self.mesh)

        xy = 2 * self.mesh.v_tex - 1
        z = torch.full((xy.shape[0], 1), 0.5, device=xy.device)
        w = torch.ones(xy.shape[0], 1, device=xy.device)
        xyzw = torch.cat([xy, z, w], dim=1)
        tri = self.mesh.t_pos_idx
        rast, _ = rasterize(self.glctx, xyzw.unsqueeze(0), tri.int(), (1024, 1024))
        gb_pos, _ = interpolate(
            self.mesh.v_pos[None, ...], rast, self.mesh.t_pos_idx.int()
        )
        self.coordmap = torch.cat([gb_pos, torch.ones_like(gb_pos[..., :1])], dim=-1)

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

        self.optimizer = torch.optim.Adam(
            [self.texture], self.cfg.learning_rate, weight_decay=self.cfg.decay
        )

    def render(self, camera, *args, **kwargs) -> torch.Tensor:
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
            *args,
            **kwargs,
        )
        image = (
            render_pkg["shaded"][..., :-1].permute(0, 3, 1, 2).contiguous()
        )  # [B, 3, H, W]
        alpha = render_pkg["shaded"][..., -1].unsqueeze(1)  # [B, 1, H, W]

        return {
            "image": image,
            "alpha": alpha.detach(),
        }

    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        """
        Render the splats to an image.

        :return: The rendered image. Shape [B, 3, H, W].
        """
        elevs = (0, 0, 0, 0, 30, 30, 30, 30)
        azims = (0, 90, 180, 270, 45, 135, 225, 315)
        dists = [sum(self.cfg.dist_range) / 2] * len(elevs)

        cameras = sm.dataset.params_to_cameras(
            dists,
            elevs,
            azims,
        )

        image = self.render(cameras)["image"]
        # encode then decode to remove artifacts
        # image = sm.prior.encode_image_if_needed(image)
        # image = sm.prior.decode_latent(image)

        return image

    def optimize(self, step: int) -> None:
        self.schedule_lr(step)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def closed_form_optimize(self, step, camera, target):
        if self.texture.shape[1] == 3:
            target = sm.prior.decode_latent_if_needed(target)
        elif self.texture.shape[1] == 4:
            target = sm.prior.encode_image_if_needed(target)

        # raise NotImplementedError("Closed form optimization not implemented for mesh model")
        num_cameras = camera["num"]

        texture_new = torch.zeros_like(self.texture)
        texture_cnt = torch.zeros_like(self.texture, dtype=torch.long)
        if self.cfg.use_selection:
            print("selection mode...")
            idx = list(range(num_cameras))
            idx = idx[step % num_cameras:] + idx[:step % num_cameras]
            # shuffle idx except the first one
            idx = [idx[0]] + np.random.permutation(idx[1:]).tolist()
            # if (step // num_cameras) % 2 == 1:
            #     idx = idx[::-1]

            print(idx)
            for i in idx:
                cam = index_camera(camera, i)
                texture, mask = unproject(cam, target[i:i+1], self.coordmap, return_mask=True)
                assert texture.dim() == 4, f"Invalid texture shape: {texture.shape}"
                texture_new[texture_cnt == 0] = texture[texture_cnt == 0]
                # round mask
                texture_cnt += mask.long()
            # texture_new = texture_new / (texture_cnt + 1e-6)
        else:
            for i in range(num_cameras):
                cam = index_camera(camera, i)
                texture, mask = unproject(cam, target[i:i+1], self.coordmap, return_mask=True)
                assert texture.dim() == 4, f"Invalid texture shape: {texture.shape}"
                texture_new += texture
                # round mask
                texture_cnt += mask.long()
            texture_new = texture_new / (texture_cnt + 1e-6)
        texture_new[texture_cnt == 0] = self.texture[texture_cnt == 0]

        self.texture.data = texture_new

    def schedule_lr(self, step):
        """
        Adjust the learning rate (optional).
        """
        return


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

        dist_range: Tuple[float, float] = (1.5, 1.5)

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

    def render(self, camera, *args, **kwargs):
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
            *args,
            **kwargs,
        )
        image = (
            render_pkg["shaded"][..., :-1].permute(0, 3, 1, 2).contiguous()
        )  # [B, 3, H, W]
        alpha = render_pkg["shaded"][..., -1].unsqueeze(1)  # [B, 1, H, W]

        return {
            "image": image,
            "alpha": alpha.detach(),
        }

    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        """
        Render the splats to an image.

        :return: The rendered image. Shape [B, 3, H, W].
        """
        elevs = (0, 0, 0, 0, 30, 30, 30, 30)
        azims = (0, 90, 180, 270, 45, 135, 225, 315)
        dists = [sum(self.cfg.dist_range) / 2] * len(elevs)

        cameras = sm.dataset.params_to_cameras(
            dists,
            elevs,
            azims,
        )

        image = self.render(cameras)["image"]
        # encode then decode to remove artifacts
        image = sm.prior.encode_image_if_needed(image)
        image = sm.prior.decode_latent(image)

        return image

    def optimize(self, step: int) -> None:
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.schedule_lr(step)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def schedule_lr(self, step):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


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

    optim = torch.optim.Adam(params, cfg.learning_rate, weight_decay=cfg.decay)
    activate_scheduler = cfg.lr_decay < 1 and cfg.decay_step > 0 and not cfg.lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=cfg.decay_step, gamma=cfg.lr_decay
        )

    return net, optim, activate_scheduler, lr_scheduler
