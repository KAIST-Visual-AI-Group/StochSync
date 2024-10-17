import os
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn 
import torch.nn.functional as F
from utils.config_utils import load_config
import shared_modules as sm
from data import DATASETs
from background import BACKGROUNDs
from model import MODELs
from prior import PRIORs
from sampler import SAMPLERs
from logger import LOGGERs
from time_sampler import TIME_SAMPLERs
from noise_sampler import NOISE_SAMPLERs
from utils.extra_utils import ignore_kwargs, get_class_filename, redirect_stdout_to_tqdm
from utils.extra_utils import redirected_tqdm as re_tqdm
from utils.extra_utils import redirected_trange as re_trange
from utils.camera_utils import merge_camera
import utils.prior_utils as pu
from utils.print_utils import print_with_box, print_info, print_warning
from utils.image_utils import save_tensor

from tqdm import tqdm, trange


# PSNR
def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


class DistillationTrainer(ABC):
    """
    Abstract base class for all trainers.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        eval_dir: str = None
        dataset: str = "random"
        background: str = "random_solid"
        model: str = "gs"
        prior: str = "sd"
        sampler: str = "sds"
        time_sampler: str = "linear_annealing"
        noise_sampler: str = "sds"
        logger: str = "simple"
        max_steps: int = 10000
        init_step: int = 0
        output: str = "output"
        prefix: str = ""
        save_source: bool = True
        recon_steps: int = 30
        initial_recon_steps: Optional[int] = 1
        recon_type: str = "rgb"
        weighting_scheme: str = "sds"  # sds, fixed
        use_closed_form: bool = True
        use_ode: bool = False
        disable_debug: bool = False

        ode_steps: int = 100
        log_interval: int = 100
        seam_removal_steps: int = 0
        loss_scale: float = 2000

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        sm.dataset = DATASETs[self.cfg.dataset](cfg_dict)
        sm.background = BACKGROUNDs[self.cfg.background](cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        sm.prior = PRIORs[self.cfg.prior](cfg_dict)
        sm.time_sampler = TIME_SAMPLERs[self.cfg.time_sampler](cfg_dict)
        sm.noise_sampler = NOISE_SAMPLERs[self.cfg.noise_sampler](cfg_dict)
        sm.logger = LOGGERs[self.cfg.logger](cfg_dict)

        if self.cfg.eval_dir is None:
            self.eval_dir = os.path.join(self.cfg.root_dir, "eval")
        else:
            self.eval_dir = self.cfg.eval_dir
        
        os.makedirs(self.cfg.root_dir, exist_ok=True)
        os.makedirs(f"{self.cfg.root_dir}/debug", exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

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
            
            from prior.base import Prior
            filename = get_class_filename(Prior)
            os.system(f"cp {filename} {self.cfg.root_dir}/src/base_prior.py")
            
            filename = get_class_filename(sm.time_sampler)
            os.system(f"cp {filename} {self.cfg.root_dir}/src/time_sampler.py")
            
            filename = get_class_filename(sm.noise_sampler)
            os.system(f"cp {filename} {self.cfg.root_dir}/src/noise_sampler.py")
            

        sm.model.prepare_optimization()

    def train_single_step(self, step: int, prev_eps=None) -> Any:
        def g(camera):
            r_pkg = sm.model.render(camera)
            bg = sm.background(camera)
            return r_pkg["image"] + bg * (1 - r_pkg["alpha"])

        # 1. Sample camera
        camera = sm.dataset.generate_sample()

        # 2. Render image
        latent = sm.prior.encode_image_if_needed(g(camera))

        # 3. Sample time
        t_curr = sm.time_sampler(step)
            
        total_loss = sm.noise_sampler(
            camera, latent, t_curr, prev_eps
        )

        recon_steps = self.cfg.initial_recon_steps
        with re_trange(
            recon_steps, position=1, desc="Regression", leave=False
        ) as pbar:
            for in_step in pbar:
                total_loss.backward()

                sm.model.optimize(step)
                if hasattr(sm.background, "optimize"):
                    sm.background.optimize(step)

                pbar.set_postfix(reg_loss=total_loss.item())
                
                
        with torch.no_grad():
            # Log the result
            if not self.cfg.disable_debug and step % self.cfg.log_interval == 0:
                images_for_log = sm.prior.decode_latent_if_needed(latent)
                sm.logger(
                    step, camera, images_for_log,
                )

                
        final_loss = total_loss.item()
        eps_pred = None 
        
        return final_loss, eps_pred

    def train(self):
        with redirect_stdout_to_tqdm():
            eps_pred = None

            with re_trange(
                self.cfg.init_step,
                self.cfg.max_steps,
                position=0,
                desc="Denoising Step",
                initial=self.cfg.init_step,
                total=self.cfg.max_steps,
            ) as pbar:
                for step in pbar:
                    loss, eps_pred = self.train_single_step(step, prev_eps=eps_pred)
                    pbar.set_postfix(loss=loss)
            sm.logger.end_logging()

            output_filename = os.path.join(
                self.cfg.root_dir, f"{self.cfg.prefix}_{self.cfg.output}"
            )
            sm.model.save(output_filename)
            sm.model.render_eval(self.eval_dir)

            return output_filename
