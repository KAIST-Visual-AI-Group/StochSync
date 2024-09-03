import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass

from utils.extra_utils import ignore_kwargs
from .base import BaseModel
from .radiance_fields.ngp import NGPColorField
from .radiance_fields.utils import Rays, render_image_with_occgrid
from nerfacc.estimators.occ_grid import OccGridEstimator
import shared_modules as sm

from k_utils.print_utils import print_info


class NGPModel(BaseModel):

    @ignore_kwargs
    @dataclass
    class Config:
        max_steps: int = 10000
        device: str = "cuda"
        width: int = 512
        height: int = 512
        model_path: Optional[str] = None

        learning_rate: float = 0.01
        near_plane: float = 0.0
        far_plane: float = 1.0e10
        render_step_size: float = 0.02
        use_density_bias: bool = True

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)

        aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.cfg.device)
        grid_resolution = 32
        grid_nlvl = 1

        self.estimator = OccGridEstimator(
            roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
        ).to(self.cfg.device)

        # self.radiance_field = NGPRadianceField(aabb=self.estimator.aabbs[-1]).to(self.cfg.device)
        self.radiance_field = NGPColorField(
            aabb=self.estimator.aabbs[-1], use_density_bias=self.cfg.use_density_bias
        ).to(self.cfg.device)

        self.optimizer = torch.optim.Adam(
            self.radiance_field.parameters(),
            lr=self.cfg.learning_rate,
            eps=1e-15,
            weight_decay=1e-6,
        )

        if self.cfg.model_path is not None:
            self.load(self.cfg.model_path)

        self.dump = {}

    def load(self, path: str) -> None:
        print(f"Loading model from {path}")
        ckpt = torch.load(path)
        estimator_state_dict = ckpt["estimator_state_dict"]
        radiance_field_state_dict = ckpt["radiance_field_state_dict"]
        optimizer_state_dict = ckpt["optimizer_state_dict"]

        self.estimator.load_state_dict(estimator_state_dict)
        self.radiance_field.load_state_dict(radiance_field_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

        print_info("warming up the estimator")
        for _ in range(10):
            self.update_estimator()

    def save(self, path: str) -> None:
        ckpt = {
            "estimator_state_dict": self.estimator.state_dict(),
            "radiance_field_state_dict": self.radiance_field.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)

    def reset_optimizer(self):
        """
        Reset the state of the optimizer and learning scheduler.
        """
        self.optimizer = torch.optim.Adam(
            self.radiance_field.parameters(),
            lr=self.cfg.learning_rate,
            eps=1e-15,
            weight_decay=1e-6,
        )

    def prepare_optimization(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Prepare the model for optimization. This might include setting up optimizers and other related tasks.

        :param parameters: Optional dictionary of parameters for optimization.
        """
        self.estimator.train()
        self.radiance_field.train()
        self.update_estimator()

    def render(self, camera) -> torch.Tensor:
        if not torch.is_grad_enabled():
            self.estimator.eval()
            self.radiance_field.eval()

        c2ws, Ks, width, height = (
            camera["c2w"],
            camera["K"],
            camera["width"],
            camera["height"],
        )

        images = []
        alphas = []
        for i in range(camera["num"]):
            # Generate rays
            origins = c2ws[i : i + 1, :3, 3]  # (1, 3)
            origins = origins.view(1, 1, 3).expand(height, width, -1)  # (H, W, 3)

            pixel_coords = (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(0, height - 1, height),
                        torch.linspace(0, width - 1, width),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                .flip([2])
                .to(Ks.device)
            )  # (H, W, 2)
            pixel_coords = pixel_coords.reshape(-1, 2)  # (H*W, 2)

            dirs = (
                Ks[i : i + 1].inverse().squeeze()
                @ torch.cat(
                    [
                        pixel_coords,
                        torch.ones(pixel_coords.shape[0], 1).to(pixel_coords.device),
                    ],
                    dim=1,
                ).T
            )  # (3, 512*512)
            dirs = dirs / torch.norm(dirs, dim=0, keepdim=True)  # (3, H*W)
            # convert to world space
            dirs = c2ws[i : i + 1, :3, :3] @ dirs.unsqueeze(0)  # (1, 3, H*W)
            dirs = dirs.squeeze(0).permute(1, 0).view(height, width, 3)  # (H, W, 3)

            # Create rays
            rays = Rays(origins=origins, viewdirs=dirs)

            # render
            rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                self.radiance_field,
                self.estimator,
                rays,
                near_plane=self.cfg.near_plane,
                render_step_size=self.cfg.render_step_size,
            )
            images.append(rgb)
            alphas.append(acc)
            # rgb: [H, W, 3], acc: [H, W, 1]
        images = torch.stack(images, dim=0).permute(0, 3, 1, 2)  # [N, 3, H, W]
        alphas = torch.stack(alphas, dim=0).permute(0, 3, 1, 2)

        if not torch.is_grad_enabled():
            self.estimator.train()
            self.radiance_field.train()

        return {
            "image": images.nan_to_num(),
            "alpha": alphas.nan_to_num(),
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
    
    def optimize(self, step: int) -> None:
        """
        Optimize the model parameters for the given step.

        :param step: Current step of the optimization process.
        """

        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.cfg.render_step_size

        self.schedule_lr(step)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
        )

    def regularize(self):
        """
        Regularize the model parameters (optional).
        """
        return torch.tensor(0.0, device=self.device)

    def schedule_lr(self, step):
        """
        Adjust the learning rate (optional).
        """
        pass

    def update_estimator(self):
        """
        Update the occupancy grid estimator (optional).
        """

        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            return density * self.cfg.render_step_size

        self.estimator.update_every_n_steps(
            step=1,
            occ_eval_fn=occ_eval_fn,
            n=1,
        )

    def render_self(self) -> torch.Tensor:
        raise NotImplementedError