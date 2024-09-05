from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .image import ImageModel

import shared_modules
from utils.extra_utils import ignore_kwargs
from utils.panorama_utils import pano_to_pers_raw, pers_to_pano_raw, pano_to_pers_accum_raw
from k_utils.image_utils import save_tensor, pil_to_torch
from k_utils.print_utils import print_warning

class PanoramaModel(ImageModel):
    """
    Model for rendering and optimizing a 2D image with sigmoid activation for each pixel.
    """

    @ignore_kwargs
    @dataclass
    class Config(ImageModel.Config):
        channels: int = 3
        pano_height: int = 2048
        pano_width: int = 4096
        initialization: str = "random"  # random, zero, gray, image
        init_img_path: Optional[str] = None

        learning_rate: float = 0.1

    def __init__(self, cfg={}):
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.image = None
        self.optimizer = None
        self.noise_map = torch.randn(1, 4, 2048, 4096, device=self.cfg.device)

    def prepare_optimization(self) -> None:
        self.image = torch.nn.Parameter(
            self.initialize_image(
                self.cfg.channels,
                self.cfg.pano_height,
                self.cfg.pano_width,
                self.cfg.initialization,
                self.cfg.init_img_path,
            )
        )
        self.optimizer = torch.optim.Adam([self.image], lr=self.cfg.learning_rate)

    # @torch.no_grad()
    # def save(self, path: str) -> None:
    #     if self.cfg.channels == 3:
    #         img = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)
    #         # crop based on gray-color search
    #         print_warning('Manually cropping panorama image before saving. This is a temporary solution.')
    #         img_col = img[0, :, :, 0]
    #         first_pixel = img_col[:, 0:1]
    #         # search the first non-gray pixel index
    #         first_non_gray_idx = torch.where(img_col != first_pixel)[-1][0]
    #         last_non_gray_idx = torch.where(img_col != first_pixel)[-1][-1]
    #         img = img[:, :, first_non_gray_idx:last_non_gray_idx + 1, :]
    #         save_tensor(img, path)
    #     elif self.cfg.channels == 4:
    #         latent = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)
    #         img = shared_modules.prior.decode_latent(latent)
    #         save_tensor(img, path)
    #     else:
    #         raise ValueError(f"Channels must be 3 or 4, got {self.cfg.channels}")
    
    def render(self, camera) -> torch.Tensor:
        num_cameras = camera["num"]
        width, height, fov, azim, elev = (
            camera["width"],
            camera["height"],
            camera["fov"],
            camera["azimuth"],
            camera["elevation"],
        )

        img_projected = []
        for i in range(num_cameras):
            img_projected.append(
                pano_to_pers_raw(
                    self.image.unsqueeze(0),
                    fov,
                    azim[i],
                    elev[i],
                    height,
                    width,
                )
            )
        img_projected = torch.cat(img_projected, dim=0)

        return {
            "image": img_projected,
            "alpha": torch.ones(num_cameras, 1, height, width, device=self.cfg.device),
        }
    
    @torch.no_grad()
    def render_self(self) -> torch.Tensor:
        image = self.image if self.image.dim() == 4 else self.image.unsqueeze(0)
        if image.shape[1] == 3:
            pass
            # latent = shared_modules.prior.encode_image(image)
            # image = shared_modules.prior.decode_latent(latent)
        elif image.shape[1] == 4:
            elevs = (0, 0, 0, 0, 0, 50, 50, 50, 50, -50, -50, -50, -50)
            azims = (0, 72, 144, 216, 288, 0, 90, 180, 270, 0, 90, 180, 270)
            dists = [1.5] * len(elevs)
            cameras = shared_modules.dataset.params_to_cameras(
                dists,
                elevs,
                azims,
            )
            latents = self.render(cameras)["image"]
            rgbs = shared_modules.prior.decode_latent(latents)

            # unprojection
            num_cameras = cameras["num"]
            width, height, fov, azim, elev = (
                cameras["width"],
                cameras["height"],
                cameras["fov"],
                cameras["azimuth"],
                cameras["elevation"],
            )

            img_new = torch.zeros(1, 3, 2048, 4096, device=self.cfg.device)
            img_cnt = torch.zeros(1, 1, 2048, 4096, device=self.cfg.device)
            print("start")
            for i in range(num_cameras):
                print(':', i)
                img_tmp, mask = pers_to_pano_raw(
                    rgbs[i:i+1],
                    fov,
                    azim[i],
                    elev[i],
                    2048,
                    4096,
                    return_mask=True,
                )
                print(':', i)
                img_new += img_tmp.squeeze(0)
                img_cnt += mask.squeeze(0).long()
            
            image = img_new / (img_cnt + 1e-6)

        return image

    @torch.no_grad()
    def closed_form_optimize(self, step, camera, target):
        if self.image.shape[0] == 3:
            target = shared_modules.prior.decode_latent_if_needed(target)
        elif self.image.shape[0] == 4:
            target = shared_modules.prior.encode_image_if_needed(target)

        num_cameras = camera["num"]
        width, height, fov, azim, elev = (
            camera["width"],
            camera["height"],
            camera["fov"],
            camera["azimuth"],
            camera["elevation"],
        )

        img_new = torch.zeros_like(self.image)
        img_cnt = torch.zeros_like(self.image, dtype=torch.long)
        print("start")
        for i in range(num_cameras):
            print(':', i)
            img_tmp, mask = pers_to_pano_raw(
                target[i:i+1],
                fov,
                azim[i],
                elev[i],
                self.cfg.pano_height,
                self.cfg.pano_width,
                return_mask=True,
            )
            print(':', i)
            img_new += img_tmp.squeeze(0)
            # round mask
            img_cnt += mask.squeeze(0).long()
        
        img_new = img_new / (img_cnt + 1e-6)
        img_new[img_cnt == 0] = self.image[img_cnt == 0]
        print("end")

        self.image.data = img_new
    

    def get_noise(self, camera):
        self.noise_map = torch.randn(1, 4, 2048, 4096, device=self.cfg.device)
        num_cameras = camera["num"]
        width, height, fov, azim, elev = (
            camera["width"],
            camera["height"],
            camera["fov"],
            camera["azimuth"],
            camera["elevation"],
        )
        width = width // 8
        height = height // 8

        noise_projected = []
        for i in range(num_cameras):
            pers, cnt = pano_to_pers_accum_raw(self.noise_map, fov, azim[i], elev[i], height, width)
            pers = pers / (torch.sqrt(cnt) + 1e-8)
            noise_projected.append(pers)
        noise_projected = torch.cat(noise_projected, dim=0)

        return noise_projected
        