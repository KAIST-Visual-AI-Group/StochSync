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
from time_sampler import TIME_SAMPLERs
from noise_sampler import NOISE_SAMPLERs
from utils.extra_utils import ignore_kwargs, get_class_filename, redirect_stdout_to_tqdm
from utils.extra_utils import redirected_tqdm as re_tqdm
from utils.extra_utils import redirected_trange as re_trange
from utils.camera_utils import merge_camera
import utils.prior_utils as pu
from k_utils.print_utils import print_with_box, print_info, print_warning
from k_utils.image_utils import save_tensor

from tqdm import tqdm, trange


class ResamplingTrainer(ABC):
    """
    Abstract base class for all trainers.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
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
        initial_recon_steps: Optional[int] = None
        recon_type: str = "rgb"
        weighting_scheme: str = "sds"  # sds, fixed
        use_closed_form: bool = True
        disable_debug: bool = False

        log_interval: int = 100
        
        # Re-Sampling 
        use_resampling: bool = False
        resampling_iter: int = 10
        resampling_method: str = "mcg"  # mcg, rejection
        reproj_error_threshold: float = 0.1
        

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        sm.dataset = DATASETs[self.cfg.dataset](cfg_dict)
        sm.background = BACKGROUNDs[self.cfg.background](cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        sm.prior = PRIORs[self.cfg.prior](cfg_dict)
        sm.time_sampler = TIME_SAMPLERs[self.cfg.time_sampler](cfg_dict)
        sm.noise_sampler = NOISE_SAMPLERs[self.cfg.noise_sampler](cfg_dict)
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

        with torch.no_grad():
            # 1. Sample camera 
            print(1)
            camera = sm.dataset.generate_sample()
            
            # 2. Render image 
            print(2)
            latent = sm.prior.encode_image_if_needed(g(camera))
            
            # 3. Sample time 
            print(3)
            t_curr = sm.time_sampler(step)
            
            best_loss = float("inf")
            for ri in range(self.cfg.resampling_iter):
                # 4. Sample noise
                print(4)
                
                # noise = sm.noise_sampler(camera, latent, t_curr, prev_eps)
                if prev_eps is None:
                    noise = torch.randn_like(latent)
                else:
                    noise = prev_eps

                # 5. Perturb-recover to get the GT latent
                print(5)
                if step == 0:
                    latent_noisy = noise
                else:
                    latent_noisy = sm.prior.add_noise(latent, t_curr, noise=noise)
                    
                noise_preds = sm.prior.predict(camera, latent_noisy, t_curr)
                gt_tweedie = sm.prior.get_tweedie(latent_noisy, noise_preds, t_curr)
                
                if self.cfg.use_resampling:
                    prev_eps = noise_preds
                    
                    reproj_error = sm.model.compute_reproj_error(gt_tweedie, camera)
                    if reproj_error < best_loss:
                        best_loss = reproj_error
                        best_tweedie = gt_tweedie
                        best_latent_noisy = latent_noisy
                        
                    print(f"reproj_error : {reproj_error}")
                    
                    if self.cfg.resampling_method == "rejection":
                        if step == 0 or reproj_error < self.cfg.reproj_error_threshold:
                            break
                
                    elif self.cfg.resampling_method == "mcg":
                        raise NotImplementedError("MCG resampling is not implemented yet.")
            
            gt_tweedie = best_tweedie
            latent_noisy = best_latent_noisy
            
            # =========================================================

            # 5.5. Calculate the weighting coefficient
            print(5.5)
            if self.cfg.weighting_scheme == "sds":
                alpha_t = sm.prior.pipeline.scheduler.alphas_cumprod[t_curr].to(latent)
                coeff = ((1 - alpha_t)*alpha_t)**0.5
            elif self.cfg.weighting_scheme == "fixed":
                coeff = 0.32  # to match the scale of the sds weighting
            else:
                raise ValueError(f"Unknown weighting scheme: {self.cfg.weighting_scheme}")
            
            # 6. Define the target image depending on the reconstruction type
            print(6)
            if self.cfg.recon_type == "rgb":
                gt_image = sm.prior.decode_latent(gt_tweedie)
                gt_image = torch.clamp(gt_image, 0.01, 0.99)
                target = gt_image
            else:
                target = gt_tweedie
                
        # 7. Optimize the rendering to match the target
        print(7)
        final_loss = 0.0
        if self.cfg.use_closed_form:
            sm.model.closed_form_optimize(step, camera, target)
        else:
            recon_steps = self.cfg.recon_steps
            if self.cfg.initial_recon_steps is not None and step == 0:
                print_info("Using another # of steps for the first step...")
                recon_steps = self.cfg.initial_recon_steps

            with re_trange(recon_steps, position=1, desc="Regression") as pbar:
                for in_step in pbar:
                    if self.cfg.recon_type == "latent":
                        source = sm.prior.encode_image_if_needed(g(camera))
                    else:
                        source = sm.prior.decode_latent_if_needed(g(camera))
                        
                    total_loss = coeff * F.mse_loss(source, target, reduction="sum") / camera["num"]
                    total_loss.backward()
                    
                    sm.model.optimize(in_step)
                    if hasattr(sm.background, "optimize"):
                        sm.background.optimize(in_step)
                        
                    pbar.set_postfix(reg_loss=total_loss.item())
            final_loss = total_loss.item()

        # 8. Calculate the pseudo noise
        print(8)
        with torch.no_grad():
            print_warning("Rendering the image again to calculate pseudo noise...")
            latent = sm.prior.encode_image_if_needed(g(camera))
            eps_pred = sm.prior.get_eps(latent_noisy, latent, t_curr)

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

            return output_filename
