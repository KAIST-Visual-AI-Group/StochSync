import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from utils.camera_utils import generate_camera_params
from typing import Literal, Tuple
from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs

from .base import InfiniteDataset
from utils.camera_utils import generate_camera, merge_camera
from abc import ABC, abstractmethod

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

    def params_to_cameras(self, dists, elevs, azims):
        cameras = []
        for dist, elev, azim in zip(dists, elevs, azims):
            camera = generate_camera(
                dist,
                elev,
                azim,
                self.cfg.fov,
                self.cfg.height,
                self.cfg.width,
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
        dists = []
        elevs = []
        azims = []
        interval = 360 / self.cfg.batch_size

        dist = np.random.uniform(*self.cfg.dist_range)
        elev = np.random.uniform(*self.cfg.elev_range)
        azim = np.random.uniform(*self.cfg.azim_range)
        dists = dists + [dist] * self.cfg.batch_size
        elevs = elevs + [elev] * self.cfg.batch_size
        azims = [(azim + i * interval) % 360 for i in range(self.cfg.batch_size)]
            
        return self.params_to_cameras(dists, elevs, azims)


class SeqTurnaroundCameraDataset(CameraDataset):
    @ignore_kwargs
    @dataclass
    class Config(CameraDataset.Config):
        num_cameras: int = 60
        batch_size: int = 1
        dist: float = 2.0
        elev: float = 30.0

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = self.Config(**cfg)
        self.azimuths = np.linspace(0, 360, self.cfg.num_cameras, endpoint=False)
        self.count = 0
        assert self.cfg.batch_size == 1, "Batch size must be 1"

    def __len__(self) -> int:
        return self.cfg.num_cameras

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        azim = self.azimuths[idx]
        return self.params_to_cameras([self.cfg.dist], [self.cfg.elev], [azim])

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
    

class AlternateMVCameraDataset(CameraDataset):
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
        self.flag = False

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dists = []
        elevs = []
        azims = []

        dist = (self.cfg.dist_range[0] + self.cfg.dist_range[1]) / 2
        elev = (self.cfg.elev_range[0] + self.cfg.elev_range[1]) / 2
        # min, mim+interval, min+2*interval, ..., min+(batch_size-1)*interval=max
        azims = np.linspace(*self.cfg.azim_range, self.cfg.batch_size, endpoint=False)
        dists = dists + [dist] * self.cfg.batch_size
        elevs = elevs + [elev] * self.cfg.batch_size

        if self.flag:
            azims = azims[1::2]
            dists = dists[1::2]
            elevs = elevs[1::2]
        else:
            azims = azims[::2]
            dists = dists[::2]
            elevs = elevs[::2]
        self.flag = not self.flag
        return self.params_to_cameras(dists, elevs, azims)