import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
import shutil

import torch

from ..utils.image_utils import save_tensor, convert_to_video, convert_to_gif
from ..utils.print_utils import print_info, print_warning, print_error
from ..utils.extra_utils import ignore_kwargs
from .. import shared_modules


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


class NullLogger(BaseLogger):
    """
    A null logger class for skipping logging
    """

    @ignore_kwargs
    @dataclass
    class Config:
        pass

    def __init__(self, cfg) -> None:
        super().__init__()
        print_info(f"NullLogger initialized.")

    def __call__(self, step, camera, images) -> None:
        pass


class SelfLogger(BaseLogger):
    @ignore_kwargs
    @dataclass
    class Config:
        root_dir: str = "./results/default"
        log_interval: int = 100
        use_encoder_decoder: bool = False

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.root_dir = self.cfg.root_dir
        self.training_dir = os.path.join(self.cfg.root_dir, "training")
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
        print_info(f"SelfLogger initialized.")
        print_info(f"Logging results to {self.root_dir}")

    def __call__(self, step, camera, images) -> None:
        if step % self.cfg.log_interval != 0:
            return
        images = shared_modules.model.render_self()
        if self.cfg.use_encoder_decoder:
            latents = shared_modules.prior.encode_image_if_needed(images)
            images = shared_modules.prior.decode_latent(latents)
        images.clip_(0, 1)
        save_tensor(
            images,
            os.path.join(self.training_dir, f"training_{step:05d}.png"),
            save_type="cat_image",
        )

    def end_logging(self):
        num_files = len(os.listdir(self.training_dir))

        if num_files == 1:
            shutil.copyfile(
                os.path.join(self.training_dir, "training_00000.png"),
                os.path.join(self.root_dir, "result.png"),
            )
        else:
            if num_files < 20:      # lerp between 1 and 4fps for 2-20 files
                fps = int(2 + (num_files - 2) * 3 / 18)
            elif num_files < 100:   # lerp between 4 and 20fps for 20-100 files
                fps = int(4 + (num_files - 20) * 16 / 80)
            elif num_files < 1000:  # lerp between 20 and 30fps for 100-1000 files
                fps = int(20 + (num_files - 100) * 10 / 900)
            else:                   # 30fps for 1000+ files
                fps = 30

            convert_to_video(
                self.training_dir,
                os.path.join(self.root_dir, "result.mp4"),
                fps=fps,
                force=True,
            )


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
        use_encoder_decoder: bool = False
        save_type: str = "cat_image"
        save_video: bool = False

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.training_dir = os.path.join(
            self.cfg.root_dir,
            f"{self.cfg.prefix}_training" if self.cfg.prefix else "training",
        )
        self.debug_dir = os.path.join(self.cfg.root_dir, f"debug")
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
            if self.cfg.use_encoder_decoder:
                latents = shared_modules.prior.encode_image_if_needed(images)
                images = shared_modules.prior.decode_latent(latents)
                
        save_tensor(
            images,
            os.path.join(self.training_dir, f"training_{step:05d}.png"),
            save_type=self.cfg.save_type,
        )

    def end_logging(self):
        if not self.cfg.save_video:
            return
        
        num_files = len(os.listdir(self.training_dir))

        if num_files == 1:
            import shutil

            shutil.copyfile(
                os.path.join(self.training_dir, "training_00000.png"),
                os.path.join(self.cfg.root_dir, "result.png"),
            )
        else:
            if num_files < 20:  # lerp between 1 and 4fps for 2-20 files
                fps = int(2 + (num_files - 2) * 3 / 18)
            elif num_files < 100:  # lerp between 4 and 20fps for 20-100 files
                fps = int(4 + (num_files - 20) * 16 / 80)
            elif num_files < 1000:  # lerp between 20 and 30fps for 100-1000 files
                fps = int(20 + (num_files - 100) * 10 / 900)
            else:  # 30fps for 1000+ files
                fps = 30

            convert_to_video(
                self.training_dir,
                os.path.join(self.cfg.root_dir, "result.mp4"),
                fps=fps,
                force=True,
            )


class SimpleLatentPreviewLogger(SimpleLogger):
    """
    A simple logger class with decoding latent
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.decode_mtx = torch.tensor(
            [
                #   R       G       B
                [0.298, 0.207, 0.208],  # L1
                [0.187, 0.286, 0.173],  # L2
                [-0.158, 0.189, 0.264],  # L3
                [-0.184, -0.271, -0.473],  # L4
            ]
        ).cuda()

        self.post_processor = lambda x: (
            (0.5 * torch.einsum("bchw,cd->bdhw", x, self.decode_mtx) + 0.5).clamp(0, 1)
        )
        print_info(
            f"Detected SimpleLatentRawLogger. Overriding post-processor for latent clipping."
        )


class SimpleLatentLogger(SimpleLogger):
    """
    A simple logger class with decoding latent
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.post_processor = shared_modules.prior.decode_latent
        print_info(
            f"Detected SimpleLatentLogger. Overriding post-processor for latent decoding."
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
        print_info(
            f"Detected RendererLatentLogger. Overriding post-processor for latent decoding."
        )
