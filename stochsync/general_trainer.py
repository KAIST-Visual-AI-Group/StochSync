import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from . import shared_modules as sm
from .data import DATASETs
from .background import BACKGROUNDs
from .model import MODELs
from .prior import PRIORs
from .logger import LOGGERs
from .time_sampler import TIME_SAMPLERs
from .noise_sampler import NOISE_SAMPLERs, SAMPLERs_REQUIRING_PREV_EPS
from .utils.extra_utils import (
    ignore_kwargs,
    get_class_filename,
    redirect_stdout_to_tqdm,
)
from .utils.extra_utils import redirected_tqdm as re_tqdm
from .utils.extra_utils import redirected_trange as re_trange
from .utils.camera_utils import merge_camera
from .utils.print_utils import print_with_box, print_info, print_warning
from .utils.image_utils import save_tensor


def downscale_min(tensor, scale_factor):
    # Get the original dimensions
    B, H, W = tensor.shape  # assuming the tensor has shape (B, H, W)

    # Ensure that the height and width are divisible by the scale factor
    assert (
        H % scale_factor == 0 and W % scale_factor == 0
    ), "Dimensions must be divisible by scale factor"

    # Reshape the tensor to group nearby pixels
    tensor_reshaped = tensor.view(
        B, H // scale_factor, scale_factor, W // scale_factor, scale_factor
    )

    # Apply the min operation along the height and width axes
    tensor_min = torch.min(tensor_reshaped, dim=4)[0]  # min along width blocks
    tensor_min = torch.min(tensor_min, dim=2)[0]  # min along height blocks

    return tensor_min


class GeneralTrainer:
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
        time_sampler: str = "linear_annealing"
        noise_sampler: str = "sds"
        logger: str = "simple"
        max_steps: int = 10000
        init_step: int = 0
        output: str = "output"
        save_source: bool = False
        recon_steps: int = 30
        initial_recon_steps: Optional[int] = None
        recon_type: str = "rgb"
        weighting_scheme: str = "sds"  # sds, fixed
        use_closed_form: bool = True
        use_ode: bool = False
        disable_debug: bool = False

        ode_steps: int = 100
        log_interval: int = 100
        seam_removal_steps: int = 0          # Number of steps to apply the soft-mask strategy.
        warmup_steps: int = 0                # Set t=999 until fully cover the canonical space. 
        try_fast_sampling: bool = False      # Use the advanced DPM++ solver if possible.
        use_decoder_inversion: bool = False  # Mitigate repetitive encoder-decoder artifacts. Suggested by https://github.com/smhongok/dec-inv

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        sm.dataset = DATASETs[self.cfg.dataset](cfg_dict)
        sm.background = BACKGROUNDs[self.cfg.background](cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        sm.prior = PRIORs[self.cfg.prior](cfg_dict)
        sm.time_sampler = TIME_SAMPLERs[self.cfg.time_sampler](cfg_dict)
        sm.noise_sampler = NOISE_SAMPLERs[self.cfg.noise_sampler](cfg_dict)
        sm.logger = LOGGERs[self.cfg.logger](cfg_dict)

        print_warning("Measuring NFE.")
        self.NFE = 0

        # original_predict = sm.prior.pipeline.unet.__class__.__call__

        # def predict_wrapper(other, x_t_stack, *args, **kwargs):
        #     self.NFE += x_t_stack.shape[0]
        #     return original_predict(other, x_t_stack, *args, **kwargs)
        # sm.prior.pipeline.unet.__class__.__call__ = predict_wrapper

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

            from .prior.base import Prior

            filename = get_class_filename(Prior)
            os.system(f"cp {filename} {self.cfg.root_dir}/src/base_prior.py")

            filename = get_class_filename(sm.time_sampler)
            os.system(f"cp {filename} {self.cfg.root_dir}/src/time_sampler.py")

            filename = get_class_filename(sm.noise_sampler)
            os.system(f"cp {filename} {self.cfg.root_dir}/src/noise_sampler.py")

        sm.model.prepare_optimization()
        self.prev_eps = None

    def train_single_step(self, step: int) -> Any:
        with torch.no_grad():
            # 1. Sample camera
            camera = sm.dataset.generate_sample()

            # 1.5. Define helper function
            bg = sm.background(camera)

            def g(camera):
                r_pkg = sm.model.render(camera)
                return r_pkg["image"] + bg * (1 - r_pkg["alpha"])

            # 2. Sample time
            t_curr = sm.time_sampler(step)
            if (
                self.cfg.use_ode
                and step >= self.cfg.max_steps - self.cfg.seam_removal_steps
            ):
                t_curr = min(int(2.0 * t_curr), 999)

            # 3. Render image
            latent = None
            if self.cfg.use_decoder_inversion:
                print_info("Using decoder inversion...")
                latent = sm.prior.encode_image_precise_if_needed(g(camera))
            else:
                latent = sm.prior.encode_image_if_needed(g(camera))

            # 4. Sample noise
            noise = sm.noise_sampler(camera, latent, t_curr, self.prev_eps)

            # 5. Perturb-recover to get the GT latent
            if step < self.cfg.warmup_steps:
                t_curr = 999
            
            if t_curr == 999:
                latent_noisy = noise
            else:
                latent_noisy = sm.prior.add_noise(latent, t_curr, noise=noise)
            latent_noisy = latent_noisy.to(sm.prior.dtype)

            print_warning("Temporary debugging images for Prof. M.H.")
            rgb_noisy = sm.prior.decode_latent(latent_noisy)
            save_tensor(rgb_noisy, f"{self.cfg.root_dir}/debug/rgb_noisy_{step}.png", save_type="cat_image", row_size=2)

            if self.cfg.use_ode:
                if step >= self.cfg.max_steps - self.cfg.seam_removal_steps:
                    print_warning("Edge-preserving sampling")
                    soft_mask = sm.model.get_diffusion_softmask(camera)
                    soft_mask = soft_mask.squeeze(1)
                    soft_mask = downscale_min(soft_mask, 8)

                    gt_tweedie = sm.prior.ddim_loop(
                        camera,
                        latent_noisy,
                        t_curr,
                        0,
                        num_steps=self.cfg.ode_steps,
                        edge_preserve=True,
                        clean=latent,
                        soft_mask=soft_mask,
                        try_fast=self.cfg.try_fast_sampling
                    )
                else:
                    gt_tweedie = sm.prior.ddim_loop(
                        camera, latent_noisy, t_curr, 0, num_steps=self.cfg.ode_steps, try_fast=self.cfg.try_fast_sampling,
                    )
            else:
                eps_pred = sm.prior.predict(camera, latent_noisy, t_curr)
                gt_tweedie = sm.prior.get_tweedie(latent_noisy, eps_pred, t_curr)

            # 5.5. Calculate the weighting coefficient
            if not self.cfg.use_closed_form:
                if self.cfg.weighting_scheme == "sds":
                    alpha_t = sm.prior.ddim_scheduler.alphas_cumprod.to(latent)[t_curr]
                    coeff = (1 - alpha_t) ** 1.5 * alpha_t**0.5
                elif self.cfg.weighting_scheme == "fixed":
                    alphas = sm.prior.ddim_scheduler.alphas_cumprod.to(latent)
                    coeffs = (1 - alphas) ** 1.5 * alphas**0.5
                    coeff = coeffs.mean()
                else:
                    raise ValueError(
                        f"Unknown weighting scheme: {self.cfg.weighting_scheme}"
                    )

            # 6. Define the target image depending on the reconstruction type
            if self.cfg.recon_type == "rgb":
                gt_image = sm.prior.decode_latent(gt_tweedie)
                gt_image = torch.clamp(gt_image, 0.01, 0.99)
                target = gt_image
            else:
                target = gt_tweedie

        if hasattr(sm.background, "cache"):
            sm.background.cache(camera, target)

        # 7. Optimize the rendering to match the target
        if self.cfg.use_closed_form:
            sm.model.closed_form_optimize(step, camera, target)
            final_loss = 0
        else:
            recon_steps = self.cfg.recon_steps
            if self.cfg.initial_recon_steps is not None and step == 0:
                recon_steps = self.cfg.initial_recon_steps

            with re_trange(
                recon_steps, position=1, desc="Regression", leave=False
            ) as pbar:
                for in_step in pbar:
                    if self.cfg.recon_type == "latent":
                        source = sm.prior.encode_image_if_needed(g(camera))
                    else:
                        source = sm.prior.decode_latent_if_needed(g(camera))

                    total_loss = (
                        coeff
                        * 0.5
                        * F.mse_loss(source, target, reduction="sum")
                        / camera["num"]
                    )

                    total_loss.backward()

                    global_step = (step * recon_steps) + in_step
                    sm.model.optimize(global_step)
                    if hasattr(sm.background, "optimize"):
                        sm.background.optimize(global_step)

                    pbar.set_postfix(reg_loss=total_loss.item())
            final_loss = total_loss.item()

        # 8. Calculate the pseudo noises
        with torch.no_grad():
            # print_info("Rendering the image again to calculate pseudo noise...")
            if sm.noise_sampler.__class__ in SAMPLERs_REQUIRING_PREV_EPS:
                image = g(camera)
                tmp_latent = sm.prior.encode_image_if_needed(image)
                self.prev_eps = sm.prior.get_eps(latent_noisy, tmp_latent, t_curr)

            # Log the result
            if not self.cfg.disable_debug and step % self.cfg.log_interval == 0:
                images_for_log = latent
                gt_images_for_log = gt_tweedie
                sm.logger(
                    step, camera, torch.cat([images_for_log, gt_images_for_log], dim=0)
                )

        return final_loss

    def train(self):
        with redirect_stdout_to_tqdm():
            with re_trange(
                self.cfg.init_step,
                self.cfg.max_steps,
                position=0,
                desc="Denoising Step",
                initial=self.cfg.init_step,
                total=self.cfg.max_steps,
            ) as pbar:
                for step in pbar:
                    loss = self.train_single_step(step)
                    pbar.set_postfix(loss=loss)
            sm.logger.end_logging()

            output_filename = os.path.join(
                self.cfg.root_dir, self.cfg.output
            )
            sm.model.save(output_filename)
            if hasattr(sm.model, "render_eval"):
                print_info("render_eval detected. Rendering the final image...")
                sm.model.render_eval(self.eval_dir)

            # save NFE under the root directory
            with open(os.path.join(self.cfg.root_dir, "NFE.txt"), "w") as f:
                f.write(f"{self.NFE}\n")

            return output_filename
        