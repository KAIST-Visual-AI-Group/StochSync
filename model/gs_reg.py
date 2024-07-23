import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from gsplat.rendering import rasterization

from .gs import GSModel


class GSModelReg(GSModel):
    """Renderer for Gaussian splats w/ Regularization."""

    class Config(GSModel.Config):
        lambda_depth: float = 0.1

    def render(self, camera, rasterize_mode="classic", render_mode="RGB+ED"):
        """
        Render the splats to an image.
        """
        c2ws, Ks, width, height = (
            camera["c2w"],
            camera["K"],
            camera["width"],
            camera["height"],
        )
        # c2ws = c2ws.unsqueeze(0)
        # Ks = Ks.unsqueeze(0)

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
        info["ED"] = render_colors[..., 3:]  # used for regularization
        self.last_info = info

        return {
            "image": render_colors.permute(0, 3, 1, 2)[:, :3],
            "alpha": render_alphas.permute(0, 3, 1, 2),
            "info": info,
        }
    def regularize(self):
        # additional regularization loss term
        reg_loss = 0.0

        # alpha loss: regularize the alpha to lower at far away from the center
        # For some reason, alpha loss dominates the sampling loss when batch_size is large
        # So we scale it down by batch_size
        depths = self.last_info["ED"]  # [B, H, W, 1]
        print(depths.shape)
        print(depths.min(), depths.max(), depths.mean())
        exit()
        return reg_loss