import os
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Optional

import torch
import tqdm

import stochsync.shared_modules as sm
from stochsync.data import DATASETs
from stochsync.background import BACKGROUNDs
from stochsync.model import MODELs
from stochsync.prior import PRIORs
from stochsync.logger import LOGGERs
from stochsync.utils.config_utils import load_config
from stochsync.utils.extra_utils import ignore_kwargs
from stochsync.utils.print_utils import print_info, print_error, print_warning


class Renderer:
    """
    Renderer class for rendering 3D shapes from different camera positions.
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        force_overwrite: bool = False
        dataset: Any = "seq_turnaround"
        background: Any = "solid"
        model: Any = "latent_mesh"
        prior: Any = "sd"
        logger: Any = "renderer"

        # Model parameters
        model_path: Optional[str] = None

        # Dataset parameters
        dist: float = 2.0
        elev: float = 30.0
        fov: float = 72
        width: int = 256
        height: int = 256
        num_cameras: int = 60

        # Logging parameters
        output: str = "rendered.mp4"
        output_type: str = "video"
        fps: int = 20

    def __init__(self, cfg_dict):
        self.cfg = self.Config(**cfg_dict)
        cfg_dict.update(asdict(self.cfg))  # Update the config dict with the default values
        os.makedirs(self.cfg.root_dir, exist_ok=True)

        output = os.path.join(self.cfg.root_dir, self.cfg.output)
        if os.path.exists(output):
            if self.cfg.force_overwrite:
                print_warning(
                    f"Output file {output} already exists. Overwriting..."
                )
            else:
                print_error(f"Output file {output} already exists. Exiting...")
                exit(1)
        sm.dataset = DATASETs[self.cfg.dataset](cfg_dict)
        sm.background = BACKGROUNDs[self.cfg.background](cfg_dict)
        sm.model = MODELs[self.cfg.model](cfg_dict)
        # sm.prior = PRIORs[self.cfg.prior](cfg_dict)
        sm.logger = LOGGERs[self.cfg.logger](cfg_dict)
        sm.model.prepare_optimization()

    @torch.no_grad()
    def __call__(self) -> Any:
        for step in tqdm.tqdm(range(self.cfg.num_cameras), desc="Rendering"):
            # Sample a camera position
            camera = sm.dataset.generate_sample()

            # Render the 3D shape from the sampled camera position
            r_pkg = sm.model.render(camera)
            bg = sm.background(camera)
            images = r_pkg["image"] + bg * (1 - r_pkg["alpha"])

            # Log the result
            sm.logger(step, camera, images)

        sm.logger.end_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, extra = parser.parse_known_args()
    
    if args.config:
        cfg = load_config(args.config, cli_args=extra)
    else:
        cfg = load_config(cli_args=extra)
    renderer = Renderer(cfg)
    renderer()
    print_info(f"Rendering complete. Output saved to {renderer.cfg.output}")
