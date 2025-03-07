import torch
from typing import Literal, Tuple
from dataclasses import dataclass
import numpy as np

from .base import InfiniteDataset
from ..utils.extra_utils import ignore_kwargs
from ..utils.camera_utils import generate_camera, merge_camera

class CameraDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        fov: float = 72
        up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z"
        convention: Literal[
            "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
        ] = "RDF"
        device: str = "cuda"
        dists: Tuple[float] = (2.0,)
        elevs: Tuple[float] = (0.0,)
        azims: Tuple[float] = (30.0,)

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        assert len(self.cfg.dists) == len(self.cfg.elevs) == len(self.cfg.azims), "Length of dists, elevs, and azims must be the same"

    def params_to_cameras(self, dists, elevs, azims, height=None, width=None, fov=None):
        height = self.cfg.height if height is None else height
        width = self.cfg.width if width is None else width
        fov = self.cfg.fov if fov is None else fov
        cameras = []
        for dist, elev, azim in zip(dists, elevs, azims):
            camera = generate_camera(
                dist,
                elev,
                azim,
                fov,
                height,
                width,
                self.cfg.up_vec,
                self.cfg.convention,
                device=self.cfg.device,
            )
            cameras.append(camera)
        
        cameras = merge_camera(cameras)

        return cameras

    def generate_sample(self):
        # return self.params_to_cameras(dists, elevs, azims)
        return self.params_to_cameras(self.cfg.dists, self.cfg.elevs, self.cfg.azims)

class RandomCameraDataset(CameraDataset):
    @ignore_kwargs
    @dataclass
    class Config(CameraDataset.Config):
        batch_size: int = 1
        dist_range: Tuple[float, float] = (1.8, 2.2)
        elev_range: Tuple[float, float] = (-5, 40)
        azim_range: Tuple[float, float] = (0, 360)

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = []
        elevs = []
        azims = []
        for _ in range(self.cfg.batch_size):
            dists.append(np.random.uniform(*self.cfg.dist_range))
            elevs.append(np.random.uniform(*self.cfg.elev_range))
            azims.append(np.random.uniform(*self.cfg.azim_range))
            
        return self.params_to_cameras(dists, elevs, azims)


class RandomMVCameraDataset(CameraDataset):
    @ignore_kwargs
    @dataclass
    class Config(CameraDataset.Config):
        batch_size: int = 1
        dist_range: Tuple[float, float] = (1.8, 2.2)
        elev_range: Tuple[float, float] = (-5, 40)
        azim_range: Tuple[float, float] = (0, 360)

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = np.random.uniform(*self.cfg.dist_range, self.cfg.batch_size)
        
        #PDF proportional to cos(elev)
        u = np.random.uniform(0, 1, self.cfg.batch_size)
        sina = np.sin(np.radians(self.cfg.elev_range[0]))
        sinb = np.sin(np.radians(self.cfg.elev_range[1]))
        elevs = np.degrees(np.arcsin((sinb - sina) * u + sina))
        
        interval = 360 / self.cfg.batch_size
        azim = np.random.uniform(*self.cfg.azim_range)
        azims = [(azim + i * interval) % 360 for i in range(self.cfg.batch_size)]
            
        return self.params_to_cameras(dists, elevs, azims)


class SeqTurnaroundCameraDataset(CameraDataset):
    @ignore_kwargs
    @dataclass
    class Config(CameraDataset.Config):
        num_cameras: int = 60
        batch_size: int = 1
        dist: float = 2.0
        elev_min: float = -10.0
        elev_max: float = 30.0

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.azimuths = np.linspace(0, 360 * 2, self.cfg.num_cameras, endpoint=False)
        self.azimuths = self.azimuths % 360

        fluctuation = (1 - np.cos(np.linspace(0, 6 * np.pi, self.cfg.num_cameras))) / 2
        self.elevations = self.cfg.elev_min + (self.cfg.elev_max - self.cfg.elev_min) * fluctuation

        self.count = 0
        assert self.cfg.batch_size == 1, "Batch size must be 1"

    def __len__(self) -> int:
        return self.cfg.num_cameras

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        azim = self.azimuths[idx]
        elev = self.elevations[idx]
        return self.params_to_cameras([self.cfg.dist], [elev], [azim])

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.__getitem__(self.count)
        self.count += 1
        if self.count == len(self):
            self.count = 0
        return data


#=====================

class FixMVCameraDataset(CameraDataset):
    @ignore_kwargs
    @dataclass
    class Config(CameraDataset.Config):
        batch_size: int = 1
        dist_range: Tuple[float, float] = (1.8, 2.2)
        elev_range: Tuple[float, float] = (-5, 40)
        azim_range: Tuple[float, float] = (0, 360)

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = []
        elevs = []
        azims = []
        interval = 360 / self.cfg.batch_size

        dist = (self.cfg.dist_range[0] + self.cfg.dist_range[1]) / 2
        elev = (self.cfg.elev_range[0] + self.cfg.elev_range[1]) / 2
        # min, mim+interval, min+2*interval, ..., min+(batch_size-1)*interval=max
        azims = np.linspace(*self.cfg.azim_range, self.cfg.batch_size, endpoint=False)
        dists = dists + [dist] * self.cfg.batch_size
        elevs = elevs + [elev] * self.cfg.batch_size
            
        return self.params_to_cameras(dists, elevs, azims)
    

# class AlternateMVCameraDataset(CameraDataset):
#     @ignore_kwargs
#     @dataclass
#     class Config(CameraDataset.Config):
#         batch_size: int = 1
#         dist_range: Tuple[float, float] = (1.8, 2.2)
#         elev_range: Tuple[float, float] = (-5, 40)
#         azim_range: Tuple[float, float] = (0, 360)

#     def __init__(self, cfg) -> None:
#         super().__init__(cfg)
#         self.cfg = self.Config(**cfg)
#         self.flag = False

#     def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         dists = []
#         elevs = []
#         azims = []

#         dist = (self.cfg.dist_range[0] + self.cfg.dist_range[1]) / 2
#         elev = (self.cfg.elev_range[0] + self.cfg.elev_range[1]) / 2
#         # min, mim+interval, min+2*interval, ..., min+(batch_size-1)*interval=max
#         azims = np.linspace(*self.cfg.azim_range, self.cfg.batch_size, endpoint=False)
#         dists = dists + [dist] * self.cfg.batch_size
#         elevs = elevs + [elev] * self.cfg.batch_size

#         if self.flag:
#             azims = azims[1::2]
#             dists = dists[1::2]
#             elevs = elevs[1::2]
#         else:
#             azims = azims[::2]
#             dists = dists[::2]
#             elevs = elevs[::2]
#         self.flag = not self.flag
#         return self.params_to_cameras(dists, elevs, azims)
    

class AlternateCameraDataset(CameraDataset):
    @ignore_kwargs
    @dataclass
    class Config(CameraDataset.Config):
        batch_size: int = 1

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.cnt = 0
        assert len(self.cfg.dists) == len(self.cfg.elevs) == len(self.cfg.azims), "Length of dists, elevs, and azims must be the same"

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = self.cfg.dists
        elevs = self.cfg.elevs
        azims = self.cfg.azims

        B = self.cfg.batch_size
        chunk = B * (self.cnt % ((len(dists) + B - 1) // B))
        azims = azims[chunk:chunk+B]
        dists = dists[chunk:chunk+B]
        elevs = elevs[chunk:chunk+B]
        self.cnt += 1
        return self.params_to_cameras(dists, elevs, azims)
    


class QuatCameraDataset(CameraDataset):
    def generate_sample(self):
        cameras = self.params_to_cameras(self.cfg.dists, self.cfg.elevs, self.cfg.azims)
        q = torch.randn(4)
        q = q / torch.norm(q)
        cameras['quat'] = q
        return cameras
    
class TorusCameraDataset(CameraDataset):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.cnt = 0

    def generate_sample(self):
        cameras = self.params_to_cameras(self.cfg.dists, self.cfg.elevs, self.cfg.azims)
        azim_offset = (self.cnt * 45) % 180
        elev_offset = (self.cnt * 60) % 180
        cameras['azimuth'] = [(azim + azim_offset) % 360 for azim in cameras['azimuth']]
        cameras['elevation'] = [(elev + elev_offset) % 360 if elev != 0 else elev for elev in cameras['elevation']]

        self.cnt += 1
        return cameras