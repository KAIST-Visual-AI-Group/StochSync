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
from k_utils.print_utils import print_with_box, print_info, print_warning
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
        init_step: int = 0
        output: str = "output"
        prefix: str = ""
        save_source: bool = True
        recon_steps: int = 30
        initial_recon_steps: Optional[int] = None
        recon_type: str = "rgb"
        use_cached_noise: bool = False
        use_closed_form: bool = True
        use_zt_noise: bool = True
        disable_debug: bool = False

        log_interval: int = 100

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

    def train_single_step(self, step: int, prev_eps=None) -> Any:
        def g(camera):
            r_pkg = sm.model.render(camera)
            bg = sm.background(camera)
            return r_pkg["image"] + bg * (1 - r_pkg["alpha"])

        # Render-Perturb-recover the images
        with torch.no_grad():
            t_curr = torch.tensor(
                min(int(1000 * (1.0 - step / self.cfg.max_steps)), 999),
                device=sm.prior.device,
            )
            camera = sm.dataset.generate_sample()

            if prev_eps is None:
                if self.cfg.use_zt_noise:
                    prev_eps = sm.model.get_noise(camera)
                else:
                    _, *res = sm.prior.latent_res
                    new_res = (camera["num"], *res)
                    prev_eps = torch.randn(*new_res, device=sm.prior.device)

            if step == 0:
                print_info("Using pure noise for the initial step...")
                perturbed = prev_eps
            else:
                latent = sm.prior.encode_image_if_needed(g(camera))
                perturbed = sm.prior.get_noisy_sample(latent, prev_eps, t_curr)

            eps_pred = sm.prior.predict(camera, perturbed, t_curr)
            gt_latent = sm.prior.get_tweedie(perturbed, eps_pred, t_curr).detach()

            if self.cfg.recon_type == "rgb":
                gt_image = sm.prior.decode_latent(gt_latent)
                gt_image = torch.clamp(gt_image, 0.01, 0.99)
                target = gt_image
            else:
                target = gt_latent

        # Try closed-form optimization first. If NotImplementedError is raised, fall back to iterative optimization.
        final_loss = 0.0
        if self.cfg.use_closed_form:
            sm.model.closed_form_optimize(step, camera, target)
        else:
            recon_steps = self.cfg.recon_steps
            if self.cfg.initial_recon_steps is not None and step == 0:
                print_info("Using another # of steps for the first step...")
                recon_steps = self.cfg.initial_recon_steps

            if hasattr(sm.model, "reset_optimizer"):
                print_info("Resetting optimizer...")
                sm.model.reset_optimizer()

            with re_trange(recon_steps, position=1, desc="Regression") as pbar:
                for in_step in pbar:
                    if self.cfg.recon_type == "latent":
                        source = sm.prior.encode_image_if_needed(g(camera))
                    else:
                        source = sm.prior.decode_latent_if_needed(g(camera))
                    total_loss = F.mse_loss(source, target, reduction="sum")
                    total_loss.backward()
                    sm.model.optimize(in_step)
                    if hasattr(sm.background, "optimize"):
                        sm.background.optimize(in_step)
                    pbar.set_postfix(reg_loss=total_loss.item())
            final_loss = total_loss.item()

        with torch.no_grad():
            print_warning("Rendering the image again to calculate pseudo noise...")
            latent = sm.prior.encode_image_if_needed(g(camera))
            eps_pred = sm.prior.get_eps(perturbed, latent, t_curr)

            # Log the result
            if not self.cfg.disable_debug and step % self.cfg.log_interval == 0:
                images_for_log = sm.prior.decode_latent(latent)
                gt_images_for_log = sm.prior.decode_latent_if_needed(target)
                sm.logger(
                    step, camera, torch.cat([images_for_log, gt_images_for_log], dim=0)
                )

        return final_loss, eps_pred

    def train(self):
        with redirect_stdout_to_tqdm():
            eps_pred = None

            if (
                sm.model.__class__.__name__ == "PanoramaModel"
                and self.cfg.use_cached_noise
            ):
                print_warning(
                    "PanoramaModel detected. Cached noise may not be effective."
                )
                camera = sm.dataset.generate_sample()
                eps_pred = sm.model.get_noise(camera)

            with re_trange(
                self.cfg.init_step,
                self.cfg.max_steps,
                position=0,
                desc="Denoising Step",
                initial=self.cfg.init_step,
                total=self.cfg.max_steps,
            ) as pbar:

                for step in pbar:
                    if self.cfg.use_cached_noise:
                        loss, eps_pred = self.train_single_step(step, prev_eps=eps_pred)
                        # if step in range(12, 22+1):
                        #     print_info(f"Forgetting noise at step {step}...")
                        #     #eps_pred = None
                        #     camera = sm.dataset.generate_sample()
                        #     eps_pred = sm.model.get_noise(camera)
                    else:
                        loss, eps_pred = self.train_single_step(step)
                    pbar.set_postfix(loss=loss)
            sm.logger.end_logging()

            output_filename = os.path.join(
                self.cfg.root_dir, f"{self.cfg.prefix}_{self.cfg.output}"
            )
            sm.model.save(output_filename)

            return output_filename
