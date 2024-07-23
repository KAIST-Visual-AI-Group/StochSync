import os
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch

from k_utils.image_utils import save_tensor, convert_to_video
from k_utils.print_utils import print_info, print_warning, print_error
from utils.extra_utils import ignore_kwargs
from utils.camera_utils import generate_camera, merge_camera
import shared_modules


class BaseLogger(ABC):
    """
    A simple abstract logger class for logging
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, camera, images) -> None:
        pass

    def get_extra_cameras(self, step):
        return []

    def end_logging(self):
        pass


class SimpleLogger(BaseLogger):
    """
    A simple logger class for logging images
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        log_interval: int = 100
        prefix: str = ""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.training_dir = os.path.join(
            self.cfg.root_dir, f"{self.cfg.prefix}_training"
        )
        self.debug_dir = os.path.join(
            self.cfg.root_dir, f"debug"
        )
        self.post_processor = lambda x: x

        os.makedirs(self.cfg.root_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)

        print_info(f"SimpleLogger initialized.")
        print_info(f"Logging results to {self.cfg.root_dir}")

    def __call__(self, step, camera, images) -> None:
        if step % self.cfg.log_interval != 0:
            return
        if isinstance(images, list):
            images = torch.cat(images, dim=0)
        
        with torch.no_grad():
            images = self.post_processor(images)
        save_tensor(
            images,
            os.path.join(self.training_dir, f"training_{step:05d}.png"),
            save_type="cat_image",
        )
    
    def log_debug(self, images, name):
        if isinstance(images, list):
            images = torch.cat(images, dim=0)

        with torch.no_grad():
            images = self.post_processor(images)

        save_tensor(
            images,
            os.path.join(self.debug_dir, f"{name}.png"),
            save_type="cat_image",
        )


class SimpleLatentLogger(SimpleLogger):
    """
    A simple logger class with decoding latent
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.post_processor = shared_modules.prior.decode_latent
        print_info(f"Detected SimpleLatentLogger. Overriding post-processor for latent decoding.")


class ProcedureLogger(BaseLogger):
    """
    A logger class for logging the procedure
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        log_interval: int = 50
        dists: tuple = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        elevs: tuple = (0, 0, 0, 0, 30, 30, 30, 30)
        azims: tuple = (0, 90, 180, 270, 45, 135, 225, 315)
        batch_size: int = 1
        width: int = 512
        height: int = 512
        prefix: str = ""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.training_dir = os.path.join(
            self.cfg.root_dir, f"{self.cfg.prefix}_training"
        )
        extra_cameras = [
            generate_camera(
                dist, elev, azim, 72, self.cfg.height, self.cfg.height, device="cuda"
            )
            for dist, elev, azim in zip(
                self.cfg.dists,
                self.cfg.elevs,
                self.cfg.azims,
            )
        ]
        self.extra_cameras = merge_camera(extra_cameras)

        os.makedirs(self.cfg.root_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)

        print_info(f"ProcedureLogger initialized.")
        print_info(f"Logging results to {self.cfg.root_dir}")

    def __call__(self, step, camera, images) -> None:
        if step % self.cfg.log_interval != 0:
            return

        if isinstance(images, list):
            images = torch.cat(images[self.cfg.batch_size:], dim=0)
        save_tensor(
            images,
            os.path.join(
                self.training_dir,
                f"training_{step:05d}.png",
            ),
            save_type="cat_image",
        )

    def get_extra_cameras(self, step):
        if step % self.cfg.log_interval == 0:
            return [self.extra_cameras]
        return []

    def end_logging(self):
        convert_to_video(
            self.training_dir,
            os.path.join(self.cfg.root_dir, f"{self.cfg.prefix}_training.mp4"),
        )


class RendererLogger(BaseLogger):
    """
    A logger class for renderer
    """

    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        output: str = "rendered.mp4"
        output_type: str = "video"
        fps: int = 20

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.images = []
        self.post_processor = lambda x: x

    def __call__(self, step, camera, images) -> None:
        if isinstance(images, list):
            images = torch.cat(images, dim=0)
        with torch.no_grad():
            images = self.post_processor(images)
        self.images.append(images)

    def end_logging(self):
        images = torch.cat(self.images, dim=0)
        save_tensor(
            images,
            os.path.join(self.cfg.root_dir, self.cfg.output),
            save_type="video",
            fps=self.cfg.fps,
        )

class LatentRendererLogger(RendererLogger):
    """
    A simple logger class with decoding latent
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.post_processor = shared_modules.prior.decode_latent
        print_info(f"Detected RendererLatentLogger. Overriding post-processor for latent decoding.")