import math
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from plyfile import PlyData, PlyElement
from tqdm import trange

from third_party.gsplat.examples.utils import (
    normalized_quat_to_rotmat,
    knn,
)

from gsplat.rendering import rasterization

from ..utils.extra_utils import ignore_kwargs
from ..utils.gaussian_utils import create_random_pcd
from ..utils.print_utils import print_info
from ..utils.camera_utils import index_camera, camera_hash
from .. import shared_modules as sm

from .base import BaseModel

class GSTextureModel(BaseModel):
    """Renderer for Gaussian splats."""

    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 30000
        model_path: str = ""
        device: str = "cuda"
        radius: float = 0.6
        scene_scale: float = 1.0
        batch_size: int = 1
        
        xyz_lr: float = 3e-4
        scale_lr: float = 5e-3
        quat_lr: float = 1e-3
        opacity_lr: float = 5e-2
        feature_lr: float = 5e-3

        # For self-rendering
        dist_range: Tuple[float, float] = (1.8, 2.2)

        recon_steps: int = 30

    def __init__(self, args=None):
        super().__init__()
        self.cfg = self.Config(**args)
        self.splats = None
        self.optimizers = None
        self.running_stats = None
        self.last_info = None

        self.load(self.cfg.model_path)
        self.weight_dict = {}

    @property
    def scene_scale(self):
        return self.cfg.scene_scale

    def __len__(self):
        return len(self.splats["means3d"])

    def load_from_pcd(self, pcd):
        """
        Load splats from a point cloud.
        :param pcd: A tensor of shape [num_splats, 3] representing the point cloud.
        """

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        N = pcd.shape[0]
        dist2_avg = (knn(pcd, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), 0.1))  # [N,]
        features = torch.zeros(N, 3)  # [N, 3]

        self.splats = torch.nn.ParameterDict(
            {
                "means3d": torch.nn.Parameter(pcd),
                "scales": torch.nn.Parameter(scales),
                "quats": torch.nn.Parameter(quats),
                "opacities": torch.nn.Parameter(opacities),
                "features": torch.nn.Parameter(features),
            }
        ).to(self.device)

        self.splats.to(self.device)

    def load(self, path):
        """
        Load splats from a PLY file.
        :param path: Path to the PLY file.
        """
        ply_data = PlyData.read(path)

        points = np.vstack(
            [ply_data["vertex"]["x"], ply_data["vertex"]["y"], ply_data["vertex"]["z"]]
        ).T
        scales = np.vstack(
            [
                ply_data["vertex"]["scale_0"],
                ply_data["vertex"]["scale_1"],
                ply_data["vertex"]["scale_2"],
            ]
        ).T
        quats = np.vstack(
            [
                ply_data["vertex"]["rot_0"],
                ply_data["vertex"]["rot_1"],
                ply_data["vertex"]["rot_2"],
                ply_data["vertex"]["rot_3"],
            ]
        ).T
        opacities = ply_data["vertex"]["opacity"]

        # Extract features
        feature_columns = [
            col
            for col in ply_data["vertex"].data.dtype.names
            if col.startswith("f_dc_")
        ]
        features = np.vstack([ply_data["vertex"][col] for col in feature_columns]).T

        # zeroing out features
        features = np.zeros_like(features)

        self.splats = torch.nn.ParameterDict(
            {
                "means3d": torch.nn.Parameter(
                    torch.tensor(points, dtype=torch.float32)
                ),
                "scales": torch.nn.Parameter(torch.tensor(scales, dtype=torch.float32)),
                "quats": torch.nn.Parameter(torch.tensor(quats, dtype=torch.float32)),
                "opacities": torch.nn.Parameter(
                    torch.tensor(opacities, dtype=torch.float32)
                ),
                "features": torch.nn.Parameter(
                    torch.tensor(features, dtype=torch.float32)
                ),
            }
        ).to(self.device)

    def save(self, path):
        """
        Save splats to a PLY file.
        :param path: Path to the PLY file.
        """
        points = self.splats["means3d"].cpu().detach().numpy()
        scales = self.splats["scales"].cpu().detach().numpy()
        quats = self.splats["quats"].cpu().detach().numpy()
        opacities = self.splats["opacities"].cpu().detach().numpy()
        features = self.splats["features"].cpu().detach().numpy()

        # Create feature columns
        feature_columns = [f"f_dc_{i}" for i in range(features.shape[1])]

        vertex_data = np.array(
            [
                tuple(
                    np.concatenate(
                        (points[i], scales[i], quats[i], [opacities[i]], features[i])
                    )
                )
                for i in range(points.shape[0])
            ],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("scale_0", "f4"),
                ("scale_1", "f4"),
                ("scale_2", "f4"),
                ("rot_0", "f4"),
                ("rot_1", "f4"),
                ("rot_2", "f4"),
                ("rot_3", "f4"),
                ("opacity", "f4"),
                *[(name, "f4") for name in feature_columns],
            ],
        )

        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el]).write(path)

    def prepare_optimization(self, parameters=None):
        device = self.device
        batch_size = self.cfg.batch_size
        print_info(f"Preparing optimization...")
        print_info(f"Batch size: {batch_size}. Adjusting learning rates...")

        # Freeze the geometry
        lr_dict = [
            # name, lr
            # ("means3d", self.cfg.xyz_lr * self.cfg.scene_scale),
            # ("scales", self.cfg.scale_lr),
            # ("quats", self.cfg.quat_lr),
            # ("opacities", self.cfg.opacity_lr),
            ("features", self.cfg.feature_lr),
        ]

        if parameters is not None:
            lr_dict = [(name, lr) for name, lr in lr_dict if name in parameters]

        self.optimizers = [
            torch.optim.Adam(
                [
                    {
                        "params": self.splats[name],
                        "lr": lr * math.sqrt(batch_size),
                        "name": name,
                    }
                ],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, lr in lr_dict
        ]

        N = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(N, device=device),  # norm of the gradient
            "count": torch.zeros(N, device=device, dtype=torch.int),
            "weights": torch.zeros(N, device=device, dtype=torch.float32),
            "render_count": 0,
        }

    def render(self, camera, depth_mode=False, rasterize_mode="classic", render_mode="RGB"):
        """
        Render the splats to an image.
        """
        if depth_mode:
            render_mode = "ED"

        camera_num, c2ws, Ks, width, height = (
            camera["num"],
            camera["c2w"],
            camera["K"],
            camera["width"],
            camera["height"],
        )

        means = self.splats["means3d"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        colors = torch.sigmoid(self.splats["features"])  # [N, D]

        for cam_idx in range(camera_num):
            c2w = c2ws[cam_idx:cam_idx + 1]
            K = Ks[cam_idx:cam_idx + 1]
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(c2ws),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=False,
                absgrad=True,
                sparse_grad=False,
                rasterize_mode=rasterize_mode,
                render_mode=render_mode,
            )
            cam_hash = camera_hash(index_camera(camera, cam_idx))
            assert info["gs_weights"].shape == (len(means),), "Invalid weights shape"
            self.weight_dict[cam_hash] = info["gs_weights"]

        # if grad enabled
        if torch.is_grad_enabled():
            info["means2d"].retain_grad()  # used for running stats
        info["alpha"] = render_alphas  # used for regularization

        self.last_info = info

        return {
            "image": render_colors.permute(0, 3, 1, 2),
            "alpha": render_alphas.permute(0, 3, 1, 2),
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

        return self.render(cameras)["image"]
    
    def optimize(self, step):
        cfg = self.cfg
        """
        Update the optimization parameters.
        """

        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            device = self.device

    def closed_form_optimize(self, step, camera, target):
        camera_num, c2ws, Ks, width, height = (
            camera["num"],
            camera["c2w"],
            camera["K"],
            camera["width"],
            camera["height"],
        )

        weight_batch = torch.zeros((camera_num, len(self.splats["means3d"])), device=self.device)
        for cam_idx in range(camera_num):
            cam = index_camera(camera, cam_idx)
            cam_hash = camera_hash(cam)
            weights = self.weight_dict[cam_hash]
            weight_batch[cam_idx] = weights
        
        # max_weight_mask
        max_weight_idx = torch.argmax(weight_batch, dim=0)

        # print the ratio of max_weight_idx each camera
        # for cam_idx in range(camera_num):
        #     mask = (max_weight_idx == cam_idx).float()
        #     print(f"mask_ratio_{cam_idx}: {mask.mean().item()}")

        for _ in trange(self.cfg.recon_steps):
            # grad_buffers = {key: torch.zeros_like(param) for key, param in self.splats.items()}

            # for cam_idx in range(camera_num):
            #     c2w = c2ws[cam_idx:cam_idx + 1]
            #     K = Ks[cam_idx:cam_idx + 1]
            #     cam = index_camera(camera, cam_idx)
            #     rendered = self.render(cam)
            #     image = rendered["image"]
            #     alpha = rendered["alpha"]

            #     # Compute loss and backpropagate
            #     loss = F.mse_loss(image, target[cam_idx:cam_idx + 1])
            #     loss.backward()

            #     with torch.no_grad():
            #         mask = (max_weight_idx == cam_idx).float()

            #         for key, param in self.splats.items():
            #             if param.grad is not None:
            #                 if len(param.shape) == 2:  # [N, D]
            #                     grad_buffers[key] += param.grad * mask.unsqueeze(-1)
            #                 else:  # [N] or other shapes
            #                     grad_buffers[key] += param.grad * mask
            
            # # Apply the accumulated gradients
            # for key, param in self.splats.items():
            #     if param.grad is not None:
            #         param.grad = grad_buffers[key]  # Set the accumulated gradients
            
            # # Perform optimization
            # self.optimize(step)
            rendered = self.render(camera)
            image = rendered["image"]
            alpha = rendered["alpha"]

            # Compute loss and backpropagate
            loss = F.mse_loss(image, target)
            loss.backward()

            # Perform optimization
            self.optimize(step)