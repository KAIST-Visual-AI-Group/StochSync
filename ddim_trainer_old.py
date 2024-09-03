import os
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from utils.config_utils import load_config
import shared_modules as sm
from data import DATASETs
from background import BACKGROUNDs
from model import MODELs
from prior import PRIORs
from sampler import SAMPLERs
from logger import LOGGERs
from utils.extra_utils import ignore_kwargs, get_class_filename, redirect_stdout_to_tqdm
from utils.extra_utils import redirected_tqdm as re_tqdm
from utils.extra_utils import redirected_trange as re_trange
from utils.camera_utils import merge_camera
import utils.prior_utils as pu
from k_utils.print_utils import print_with_box, print_info
from k_utils.image_utils import save_tensor

from tqdm import tqdm, trange


class DDIMTrainer(ABC):
    """
    Abstract base class for all trainers.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        dataset: Any = "random"
        background: Any = "random_solid"
        model: Any = "gs"
        prior: Any = "sd"
        sampler: Any = "sds"
        logger: Any = "procedure"
        max_steps: int = 10000
        output: str = "output.ply"
        prefix: str = ""
        save_source: bool = False
        recon_steps: int = 30
        initial_recon_steps: Optional[int] = None
        recon_type: str = "rgb"
        guidance_scale: float = 100.0
        use_cached_noise: bool = False

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        sm.dataset = DATASETs[self.cfg.dataset](cfg_dict)
        sm.background = BACKGROUNDs[self.cfg.background](cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        sm.prior = PRIORs[self.cfg.prior](cfg_dict)
        sm.logger = LOGGERs[self.cfg.logger](cfg_dict)

        os.makedirs(self.cfg.root_dir, exist_ok=True)
        os.makedirs(f"{self.cfg.root_dir}/debug", exist_ok=True)

        if self.cfg.save_source:
            os.makedirs(f"{self.cfg.root_dir}/src", exist_ok=True)
            for module in [
                sm.dataset,
                sm.background,
                sm.model,
                sm.prior,
                sm.logger,
            ]:
                filename = get_class_filename(module)
                os.system(f"cp {filename} {self.cfg.root_dir}/src/")

        sm.model.prepare_optimization()

    def train_single_step(self, step: int, noise=None) -> Any:
        camera = sm.dataset.generate_sample()

        # Render-Perturb-recover the images
        with torch.no_grad():
            r_pkg = sm.model.render(camera)
            bg = sm.background(camera)
            images = r_pkg["image"] + bg * (1 - r_pkg["alpha"])

            latent = sm.prior.encode_image_if_needed(images)

            timestep = torch.tensor(
                min(int(1000 * (1.0 - step / self.cfg.max_steps)), 999)
            )

            if noise is None:
                # noise = torch.randn_like(latent)
                noise = torch.randn(4, 64*sm.model.cfg.yscale, 64*sm.model.cfg.xscale, device=sm.model.device)
                noise_list = []
                yoffsets = torch.tensor([0, 0, 0, 0, 0, 0, 0])
                xoffsets = torch.tensor([0, 32, 64, 96, 128, 160, 192])
                for i in range(7):
                    noise_list.append(
                        noise[
                            :,
                            yoffsets[i] : yoffsets[i] + 64,
                            xoffsets[i] : xoffsets[i] + 64,
                        ]
                    )
                noise = torch.stack(noise_list, dim=0)

            if step == 0:
                print_info("Using pure noise for the initial step...")
                perturbed = noise
            else:
                # perturbed = pu.get_noisy_sample(sm.prior.pipeline, latent, noise, timestep)
                perturbed = sm.prior.add_noise(latent, timestep, noise)

            eps_pred = sm.prior.predict(
                camera, perturbed, timestep.to("cuda"), self.cfg.guidance_scale
            )
            # gt_latent = pu.get_tweedie(sm.prior.pipeline, perturbed, eps_pred, timestep).detach()
            gt_latent = sm.prior.get_tweedie(perturbed, eps_pred, timestep).detach()

            if self.cfg.recon_type == "rgb" and images.shape[1] == 3:
                gt_images = sm.prior.decode_latent(gt_latent)
                gt_images = torch.clamp(gt_images, 0.01, 0.99)
            else:
                gt_images = gt_latent

        # Try closed-form optimization first. If NotImplementedError is raised, fall back to iterative optimization.
        final_loss = 0.0
        try:
            sm.model.closed_form_optimize(step, camera, gt_images)
        except NotImplementedError:
            recon_steps = self.cfg.recon_steps
            if self.cfg.initial_recon_steps is not None and step == 0:
                print_info("Using another # of steps for the first step...")
                recon_steps = self.cfg.initial_recon_steps

            # if sm.model has reset_optimizer, call it
            if hasattr(sm.model, "reset_optimizer"):
                print_info("Resetting optimizer...")
                sm.model.reset_optimizer()

            with re_trange(recon_steps, position=1, desc="Regression", leave=False) as pbar:
                for in_step in pbar:
                    r_pkg = sm.model.render(camera)
                    bg = sm.background(camera)
                    images = r_pkg["image"] + bg * (1 - r_pkg["alpha"])
                    if self.cfg.recon_type == "latent" and images.shape[1] == 3:
                        cur_images = sm.prior.encode_image(images)
                    else:
                        cur_images = images
                    total_loss = F.mse_loss(cur_images, gt_images, reduction="sum")
                    total_loss.backward()
                    sm.model.optimize(in_step)
                    # if sm.background has function optimize, call it
                    if hasattr(sm.background, "optimize"):
                        sm.background.optimize(in_step)
                    pbar.set_postfix(reg_loss=total_loss.item())
                final_loss = total_loss.item()

        if step % (max(1, self.cfg.max_steps // 100)) == 0:
            images_for_log = (
                sm.prior.decode_latent(sm.prior.encode_image(images))
                if images.shape[1] == 3
                else sm.prior.decode_latent(images)
            )
            gt_images_for_log = (
                gt_images
                if gt_images.shape[1] == 3
                else sm.prior.decode_latent(gt_images)
            )
            save_tensor(
                torch.cat([images_for_log, gt_images_for_log], dim=0),
                os.path.join(
                    f"{self.cfg.root_dir}/debug",
                    f"original_{step:05d}.png",
                ),
                save_type="cat_image",
                row_size=2,
            )

        # Log the result
        sm.logger(step, camera, images)

        return final_loss, eps_pred

    def train(self):
        with redirect_stdout_to_tqdm():
            eps_pred = None
            with re_trange(
                self.cfg.max_steps, position=0, desc="Denoising Step"
            ) as pbar:
                for step in pbar:
                    if self.cfg.use_cached_noise:
                        loss, eps_pred = self.train_single_step(step, noise=eps_pred)
                    else:
                        loss, eps_pred = self.train_single_step(step)
                    pbar.set_postfix(loss=loss)

            sm.logger.end_logging()

            output_filename = os.path.join(
                self.cfg.root_dir, f"{self.cfg.prefix}_{self.cfg.output}"
            )
            sm.model.save(output_filename)

            return output_filename
