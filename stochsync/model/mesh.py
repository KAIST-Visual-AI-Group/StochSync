from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import trimesh
import cv2

from ..utils.extra_utils import ignore_kwargs
from ..utils.image_utils import save_tensor, pil_to_torch
from ..utils.print_utils import print_info, print_warning
from ..utils.panorama_utils import remap, remap_min
from ..utils.camera_utils import index_camera
from ..utils.mesh_utils import read_obj
from ..utils.extra_utils import calculate_distance_to_zero_level
from .. import shared_modules as sm

from nvdiffrast.torch import rasterize, interpolate  # low-level rasterization functions
from .nvdiff_render.mesh import *
from .nvdiff_render.render import *
from .nvdiff_render.texture import *
from .nvdiff_render.material import *
from .nvdiff_render.obj import *
from .base import BaseModel


def normalize_mesh(v):
    center = v.mean(dim=0)
    v = v - center
    scale = torch.max(torch.norm(v, p=2, dim=1))
    v = v / scale
    return v


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

        dist_range: Tuple[float, float] = (2.5, 2.5)

        use_selection: bool = False
        texture_path: str = ""
        flip_texture: bool = False
        # max_steps: int = 10000
        # force_optim_steps: int = 0

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
        obj = read_obj(path)
        v_pos, f_idx, v_uv = obj["v"], obj["f"], obj["vt"]
        v_pos = torch.tensor(v_pos, device=self.cfg.device).float()
        f_idx = torch.tensor(f_idx, device=self.cfg.device).to(torch.int64) - 1
        v_uv = torch.tensor(v_uv, device=self.cfg.device).float()
        # f_idx, v_pos, v_uv = load_obj_uv(obj_path=path, device=self.cfg.device)
        v_pos = normalize_mesh(v_pos)
        self.mesh = Mesh(v_pos, f_idx, v_tex=v_uv, t_tex_idx=f_idx)
        # print_warning(f"Temporarily disable unit_size.")
        # self.mesh = unit_size(self.mesh)
        self.mesh = auto_normals(self.mesh)
        self.mesh = compute_tangents(self.mesh)

        xy = 2 * self.mesh.v_tex - 1
        z = torch.full((xy.shape[0], 1), 0.5, device=xy.device)
        w = torch.ones(xy.shape[0], 1, device=xy.device)
        xyzw = torch.cat([xy, z, w], dim=1)
        tri = self.mesh.t_pos_idx
        rast, _ = rasterize(
            self.glctx,
            xyzw.unsqueeze(0),
            tri.int(),
            (self.cfg.texture_size, self.cfg.texture_size),
        )
        gb_pos, _ = interpolate(
            self.mesh.v_pos[None, ...], rast, self.mesh.t_pos_idx.int()
        )
        gb_normal, _ = interpolate(
            self.mesh.v_nrm[None, ...], rast, self.mesh.t_pos_idx.int()
        )
        self.coordmap = torch.cat([gb_pos, torch.ones_like(gb_pos[..., :1])], dim=-1)
        self.normalmap = gb_normal  # [B, H, W, 3], -1~1
        self.facemap = rast[..., 3].long()
        self.maskmap = self.facemap > 0
        self.max_cosmap = torch.zeros_like(self.facemap)

    @torch.no_grad()
    def save(self, path: str) -> None:
        image = sm.prior.decode_latent_fast_if_needed(self.texture)
        save_tensor(image, path)

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
        elif self.cfg.initialization == "image":
            image = Image.open(self.cfg.texture_path).convert("RGB")
            image = pil_to_torch(image).to(self.cfg.device)
            # vertical flip
            if self.cfg.flip_texture:
                image = image.flip(2)
            self.texture = image
            assert image.dim() == 4, f"Image must have 4 dimensions, got {image.shape}"
        else:
            raise ValueError(f"Invalid initialization: {self.cfg.initialization}")

        self.optimizer = torch.optim.Adam(
            [self.texture], self.cfg.learning_rate, weight_decay=self.cfg.decay
        )

    def render(self, camera, texture=None, *args, **kwargs) -> torch.Tensor:
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

        if texture is not None:
            texture = texture.permute(0, 2, 3, 1)  # .clamp(0, 1)
        else:
            texture = self.texture.permute(0, 2, 3, 1)  # .clamp(0, 1)
        pred_material = Material(
            {
                "bsdf": "kd",
                "kd": Texture2D(texture),
                "ks": Texture2D(torch.full_like(texture, 0.5)),
                # "n": Texture2D(self.normalmap),
            }
        )

        self.mesh.material = pred_material

        if "filter_mode" not in kwargs:
            kwargs["filter_mode"] = self.cfg.sampling_mode

        lgt = None
        if "lgt" in kwargs:
            lgt = kwargs["lgt"]
            del kwargs["lgt"]

        render_pkg = render_mesh(
            self.glctx,
            self.mesh,
            mvp,  # B 4 4
            campos,  # B 3
            lgt,
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
    def render_eval(self, path) -> torch.Tensor:
        elevs = [11, 7, 7, 9, 6, 24, 25, 5, 14, 26]
        azims = [25, 32, 84, 132, 138, 144, 147, 232, 295, 355]

        dists = [2.0] * len(elevs)
        cameras = sm.dataset.params_to_cameras(
            dists,
            elevs,
            azims,
        )
        render_pkg = self.render(cameras)
        images = render_pkg["image"]
        alphas = render_pkg["alpha"]
        images = images + (1 - alphas) * 1.0
        # latents = sm.prior.encode_image_if_needed(images)
        # images = sm.prior.decode_latent(latents)
        # images.clip_(0, 1)

        fns = [f"{azi}_{_i}" for _i, azi in enumerate(azims)]
        # Save perspective view images 09.10
        save_tensor(images, path, fns=fns)

    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        """
        Render the splats to an image.

        :return: The rendered image. Shape [B, 3, H, W].
        """
        elevs = (0, 0, 0, 0, 30, 30, 30, 30)
        azims = (0, 90, 180, 270, 45, 135, 225, 315)
        dists = [2.0] * len(elevs)

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

        num_cameras = camera["num"]
        texture_new = torch.zeros_like(self.texture)
        if self.cfg.use_selection:
            max_cosmap = torch.zeros_like(self.texture, dtype=torch.float32)
            for i in range(num_cameras):
                cam = index_camera(camera, i)
                mask_dict = self.get_masks(cam)
                gp_pos_xy = mask_dict["gp_pos_xy"]
                cosmap = mask_dict["cosmap"].abs()
                mask = mask_dict["mask_depth"] & self.maskmap
                texture = self.unproject_happy(cam, target[i : i + 1], gp_pos_xy, mask)

                local_max_region = mask & (cosmap >= max_cosmap)
                texture_new[local_max_region] = texture[local_max_region]

                valid_cosmap = mask.float() * cosmap
                max_cosmap = torch.max(max_cosmap, valid_cosmap)
                self.max_cosmap = torch.max(self.max_cosmap, valid_cosmap)

            local_softmask = torch.clamp((max_cosmap - 0.1) / (0.5 - 0.1), 0, 1)
            global_mask = (self.max_cosmap == max_cosmap).float()
            softmask = torch.max(local_softmask, global_mask)
            texture_new = texture_new * softmask + self.texture * (1 - softmask)
        else:
            acc_softmask = torch.zeros_like(self.texture, dtype=torch.float32)
            for i in range(num_cameras):
                cam = index_camera(camera, i)
                mask_dict = self.get_masks(cam)
                gp_pos_xy = mask_dict["gp_pos_xy"]
                cosmap = mask_dict["cosmap"].abs()
                mask = mask_dict["mask_depth"] & self.maskmap
                texture = self.unproject(cam, target[i : i + 1], gp_pos_xy, mask)

                softmask = mask.float() * cosmap.clamp(0, 1)
                texture_new += texture * softmask
                acc_softmask += softmask

            blank_mask = acc_softmask < 1e-6
            texture_new[~blank_mask] = (texture_new / acc_softmask)[~blank_mask]
            texture_new[blank_mask] = self.texture[blank_mask]

        # Outer padding to avoid edge artifacts
        fg_mask = self.maskmap.unsqueeze(1).expand_as(texture_new)
        for _ in range(5):
            # avg
            inflated_texture = F.avg_pool2d(
                texture_new * fg_mask.float(), kernel_size=5, padding=2, stride=1
            )
            # avg cnt
            inflated_cnt = F.avg_pool2d(
                fg_mask.float(), kernel_size=5, padding=2, stride=1
            )
            texture_new[~fg_mask] = inflated_texture[~fg_mask] / (
                inflated_cnt[~fg_mask] + 1e-6
            )
            fg_mask = inflated_cnt > 0

        self.texture.data = texture_new

    def schedule_lr(self, step):
        """
        Adjust the learning rate (optional).
        """
        return

    def get_masks(self, camera):
        azims, elevs, dists = camera["azimuth"], camera["elevation"], camera["dist"]
        camera = sm.dataset.params_to_cameras(
            dists,
            elevs,
            azims,
            height=2048,
            width=2048,
        )
        c2ws, width, height, fov = (
            camera["c2w"],
            camera["width"],
            camera["height"],
            camera["fov"],
        )
        proj_mtx = util.perspective(fov * np.pi / 180, width / height, 0.1, 1000.0).to(
            c2ws.device
        )
        mvp = proj_mtx @ c2ws.inverse()

        gb_pos_clip = torch.einsum(
            "bhwp,bpq->bhwq", self.coordmap, mvp.permute(0, 2, 1)
        )
        gb_pos_clip[..., :2] = gb_pos_clip[..., :2] / gb_pos_clip[..., 3:4]
        gp_pos_xy = 0.5 * gb_pos_clip[..., :2] + 0.5

        # Cosine mask
        raydirmap = c2ws[:, :3, 3].unsqueeze(1).unsqueeze(1) - self.coordmap[..., :3]
        raydirmap = F.normalize(raydirmap, dim=-1)
        cosmap = torch.sum(raydirmap * self.normalmap, dim=-1)

        # Depth mask
        gp_pos_z = gb_pos_clip[..., 2]
        render_pkg = sm.model.render(camera, bsdf="depth")
        front_z = render_pkg["image"][:, :1]
        bg = render_pkg["alpha"] < 1.0
        front_z[bg] = front_z.max()
        gp_front_z = remap_min(front_z, gp_pos_xy * height).squeeze(1)
        mask_depth = gp_pos_z <= gp_front_z + 0.06 * torch.clamp(
            (1 - cosmap), min=0.1, max=1.0
        )

        # Face mask
        face_ids = sm.model.render(camera, bsdf="faceid")["image"][:, :1]
        valid_faces = torch.unique(face_ids.long())[1:]  # exclude background
        mask_face = torch.isin(self.facemap, valid_faces, assume_unique=False)

        return {
            "gp_pos_xy": gp_pos_xy,
            "mask_depth": mask_depth,
            "mask_face": mask_face,
            "cosmap": cosmap,
        }

    def get_diffusion_softmask(self, camera):
        masks = self.get_masks(camera)
        cosmaps = masks["cosmap"] * masks["mask_depth"]  # B H W
        cosmaps = torch.cat(
            [torch.full_like(cosmaps[:1], 0.01), cosmaps], dim=0
        )  # B+1 H W
        cosmask_idx = torch.argmax(cosmaps, dim=0)  # H W
        texture = (
            cosmask_idx.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).float()
        )  # 1 3 H W

        render_pkg = sm.model.render(camera, texture, disable_aa=True)
        cosmask_idx_view = render_pkg["image"][:, 0]
        alpha = render_pkg["alpha"][:, 0]
        indices = (
            torch.arange(1, camera["num"] + 1, device=cosmask_idx_view.device)
            .view(camera["num"], 1, 1)
            .expand_as(cosmask_idx_view)
        )
        cosmask_view = (cosmask_idx_view == indices) | (cosmask_idx_view == 0)  # B H W

        cosmask_view = (
            (~cosmask_view).cpu().numpy().astype(np.uint8)
        )  # Convert to uint8 for OpenCV
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for i in range(len(cosmask_view)):
            cosmask_view[i] = cv2.morphologyEx(cosmask_view[i], cv2.MORPH_OPEN, kernel)
        cosmask_view = torch.from_numpy(cosmask_view).bool().to(cosmask_idx_view.device)
        cosmask_view |= alpha == 0

        cosdist = calculate_distance_to_zero_level(cosmask_view)
        for i in range(len(cosdist)):
            cosdist[i] = (
                (cosdist[i] - cosdist[i].min())
                / (cosdist[i].max() - cosdist[i].min())
                * 1.7
            )
        cosdist = cosdist.clamp(0, 1)
        cosdist[alpha == 0] = 1.0

        return cosdist

    def unproject(self, camera, target, gp_pos_xy, mask):
        target_h, target_w = target.shape[-2:]
        unproj_texture = remap(target, gp_pos_xy.squeeze() * target_h, mode="nearest")

        return unproj_texture

    def unproject_happy(self, camera, target, gp_pos_xy, mask):
        target_h, target_w = target.shape[-2:]
        unproj_texture = remap(target, gp_pos_xy.squeeze() * target_h, mode="bilinear")
        orig_texture = self.texture.clone().detach()
        self.texture.data = unproj_texture * mask.float() + orig_texture * (
            1 - mask.float()
        )

        # Further optimization
        first_loss, last_loss = 0, 0
        # reset optimizer
        self.optimizer = torch.optim.Adam(
            [self.texture], self.cfg.learning_rate, weight_decay=self.cfg.decay
        )
        for i in range(20):
            r_pkg = self.render(camera)
            image = r_pkg["image"]
            alpha = r_pkg["alpha"]
            tgt = target * alpha
            loss = F.mse_loss(image, tgt)
            # if i == 0:
            #     first_loss = loss.item()
            # if i == 9:
            #     last_loss = loss.item()
            loss.backward()
            self.optimize(0)
        # print_info(f"First loss: {first_loss:.4f}, Last loss: {last_loss:.4f}")

        unproj_texture = self.texture.clone()
        self.texture.data = orig_texture

        return unproj_texture
