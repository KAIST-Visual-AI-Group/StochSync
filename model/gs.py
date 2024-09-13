import math
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch import Tensor

from third_party.gsplat.examples.utils import (
    normalized_quat_to_rotmat,
    knn,
)

from gsplat.rendering import rasterization

from .base import BaseModel
from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs
from utils.gaussian_utils import create_random_pcd
import shared_modules as sm

from utils.print_utils import print_info


class GSModel(BaseModel):
    """Renderer for Gaussian splats."""

    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 30000
        model_path: Optional[str] = None
        device: str = "cuda"
        radius: float = 0.6
        init_splats: int = 5000
        max_splats: int = 300000
        scene_scale: float = 1.0
        batch_size: int = 1
        prune_opa: float = 0.005
        grow_grad2d: float = 0.01
        grow_scale3d: float = 0.01
        prune_scale3d: float = 0.1
        refine_start_iter: int = 1500
        refine_stop_iter: int = 30000
        reset_every: int = 3000
        refine_every: int = 500
        
        xyz_lr: float = 3e-4
        scale_lr: float = 5e-3
        quat_lr: float = 1e-3
        opacity_lr: float = 5e-2
        feature_lr: float = 5e-3

        # For self-rendering
        dist_range: Tuple[float, float] = (1.8, 2.2)

    def __init__(self, args=None):
        super().__init__()
        self.cfg = self.Config(**args)
        self.splats = None
        self.optimizers = None
        self.running_stats = None
        self.last_info = None

        if self.cfg.model_path is not None:
            self.load(self.cfg.model_path)
        else:
            pcd = create_random_pcd(self.cfg.radius, self.cfg.init_splats)
            self.load_from_pcd(pcd)

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

        lr_dict = [
            # name, lr
            ("means3d", self.cfg.xyz_lr * self.cfg.scene_scale),
            ("scales", self.cfg.scale_lr),
            ("quats", self.cfg.quat_lr),
            ("opacities", self.cfg.opacity_lr),
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

    def render(self, camera, rasterize_mode="classic", render_mode="RGB"):
        """
        Render the splats to an image.
        """
        c2ws, Ks, width, height = (
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

        # if grad enabled
        if torch.is_grad_enabled():
            info["means2d"].retain_grad()  # used for running stats
        info["alpha"] = render_alphas  # used for regularization

        self.last_info = info

        return {
            "image": render_colors.permute(0, 3, 1, 2),
            "alpha": render_alphas.permute(0, 3, 1, 2),
            "info": info,
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

            if self.last_info is not None:
                self.update_running_stats(self.last_info)

            batch_adjust = 1  # int(math.sqrt(cfg.batch_size))
            refine_start_iter = cfg.refine_start_iter // batch_adjust
            refine_stop_iter = cfg.refine_stop_iter // batch_adjust
            refine_every = cfg.refine_every // batch_adjust
            if (
                step >= refine_start_iter
                and step < refine_stop_iter
                and step % refine_every == 0
            ):
                if len(self) <= self.cfg.max_splats:  # grow GSs
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    # grow GSs
                    # is_grad_high = grads >= cfg.grow_grad2d
                    # rather, sample min(size, 10000) GSs above minimum gradients(0.00001)
                    _, topk_idx = torch.topk(grads, min(len(self), 10000))
                    grad_mask = torch.zeros_like(grads, dtype=torch.bool)
                    grad_mask[topk_idx] = True
                    is_grad_high = grads >= 0.00001
                    is_grad_high = is_grad_high & grad_mask

                    is_small = (
                        torch.exp(self.splats["scales"]).max(dim=-1).values
                        <= cfg.grow_scale3d * self.scene_scale
                    )
                    is_dupli = is_grad_high & is_small
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small
                    is_split = torch.cat(
                        [
                            is_split,
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    # print(
                    #     f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    #     f"Now having {len(self)} GSs."
                    # )
                else:  # prune GSs
                    # resample GSs accorging to weights
                    prob = self.running_stats["weights"]
                    # if sum is too small, just uniformly sample
                    if prob.sum() < 1e-6:
                        prob = torch.ones_like(prob, device=device)
                    else:
                        prob /= prob.sum()
                    assert torch.all(0 <= prob) and torch.all(prob <= 1)
                    # upper bound is the number of nonzero prob entries
                    sample_size = min(
                        len(self), self.cfg.max_splats, prob.nonzero().size(0)
                    )
                    indices = np.random.choice(
                        len(self), sample_size, replace=False, p=prob.cpu().numpy()
                    )
                    mask = torch.zeros(len(self), device=device, dtype=torch.bool)
                    mask[indices] = True
                    self.refine_keep(mask)
                    # print(
                    #     f"Step {step}: Resampled GSs. " f"Now having {len(self)} GSs."
                    # )

                    # prune GSs
                    is_prune = torch.sigmoid(self.splats["opacities"]) < cfg.prune_opa
                    if step > cfg.reset_every:
                        is_too_big = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)
                    # print(
                    #     f"Step {step}: {n_prune} GSs pruned. "
                    #     f"Now having {len(self)} GSs."
                    # )

                # reset running stats
                self.running_stats["grad2d"].zero_()
                self.running_stats["count"].zero_()
                self.running_stats["weights"].zero_()
                self.running_stats["render_count"] = 0

    @torch.no_grad()
    def update_running_stats(self, info: Dict):
        """Update running stats."""
        if not hasattr(info["means2d"], "absgrad") and not hasattr(
            info["means2d"], "grad"
        ):
            return

        absgrad = True

        # normalize grads to [-1, 1] screen space
        if absgrad and hasattr(info["means2d"], "absgrad"):
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0
        grads[..., 1] *= info["height"] / 2.0
        # grads is [C, N, 2]
        sel = info["radii"] > 0.0  # [C, N]
        gs_ids = torch.where(sel)[1]  # [nnz]
        self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
        self.running_stats["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids).int())
        self.running_stats["weights"] += info["gs_weights"]
        self.running_stats["render_count"] += 1

    @torch.no_grad()
    def reset_running_stats(self):
        # reset running stats
        self.running_stats["grad2d"].zero_()
        self.running_stats["count"].zero_()
        self.running_stats["weights"].zero_()
        self.running_stats["render_count"] = 0

    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None or type(v) != torch.Tensor:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if type(v) != torch.Tensor:
                continue
            self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if type(v) != torch.Tensor:
                continue
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()
