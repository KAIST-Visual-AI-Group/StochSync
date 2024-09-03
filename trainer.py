import os
import argparse
import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from utils.config_utils import load_config
import shared_modules as sm
from data import DATASETs
from background import BACKGROUNDs
from model import MODELs
from prior import PRIORs
from sampler import SAMPLERs
from logger import LOGGERs
from utils.extra_utils import ignore_kwargs, get_class_filename
from utils.camera_utils import merge_camera
from k_utils.print_utils import print_with_box, print_info
from k_utils.image_utils import save_tensor


class Trainer(ABC):
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
        save_source: bool = True
        disable_debug: bool = False

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        sm.dataset = DATASETs[self.cfg.dataset](cfg_dict)
        sm.background = BACKGROUNDs[self.cfg.background](cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        sm.prior = PRIORs[self.cfg.prior](cfg_dict)
        sm.sampler = SAMPLERs[self.cfg.sampler](cfg_dict)
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
                sm.sampler,
                sm.logger,
            ]:
                filename = get_class_filename(module)
                os.system(f"cp {filename} {self.cfg.root_dir}/src/")

        sm.model.prepare_optimization()

    def train_single_step(self, step: int) -> Any:
        # Sample a camera position
        camera = sm.dataset.generate_sample()

        # Render the model and the background
        r_pkg = sm.model.render(camera)
        bg = sm.background(camera)
        images = r_pkg["image"] * r_pkg["alpha"] + bg * (1 - r_pkg["alpha"])

        # Sample the score and calculate the loss
        opt_loss = sm.sampler(camera, images, step)
        reg_loss = sm.model.regularize()
        total_loss = opt_loss + reg_loss

        # Backpropagate the loss and optimize the model
        total_loss.backward()
        sm.model.optimize(step)

        # Log the result
        if not self.cfg.disable_debug:
            sm.logger(step, camera, images)

        return total_loss

    def train(self):
        pbar = tqdm.tqdm(range(self.cfg.max_steps))
        for step in pbar:
            loss = self.train_single_step(step)
            pbar.set_description(f"Loss: {loss.item()}")

        sm.logger.end_logging()

        output_filename = os.path.join(
            self.cfg.root_dir, f"{self.cfg.prefix}_{self.cfg.output}"
        )
        sm.model.save(output_filename)

        return output_filename
