import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from utils.camera_utils import generate_camera_params
from typing import Literal, Tuple
from dataclasses import dataclass
from utils.extra_utils import ignore_kwargs

from .base import InfiniteDataset


class RandomCameraDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        dist_range: Tuple[float, float] = (1.8, 2.2)
        elev_range: Tuple[float, float] = (-5, 40)
        azim_range: Tuple[float, float] = (0, 360)
        fov: float = 72
        up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z"
        convention: Literal[
            "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
        ] = "RDF"
        batch_size: int = 1
        device: str = "cuda"

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        c2ws, Ks, elevs, azims = [], [], [], []
        for _ in range(self.cfg.batch_size):
            dist = np.random.uniform(*self.cfg.dist_range)
            elev = np.random.uniform(*self.cfg.elev_range)
            azim = np.random.uniform(*self.cfg.azim_range)
            c2w, K = generate_camera_params(
                dist,
                elev,
                azim,
                self.cfg.fov,
                self.cfg.width,
                self.cfg.height,
                self.cfg.up_vec,
                self.cfg.convention,
            )
            c2w = c2w.to(self.cfg.device)
            K = K.to(self.cfg.device)
            c2ws.append(c2w)
            Ks.append(K)
            elevs.append(elev)
            azims.append(azim)

        c2ws = torch.stack(c2ws)
        Ks = torch.stack(Ks)

        return {
            "batch_size": self.cfg.batch_size,
            "c2w": c2ws,
            "K": Ks,
            "width": self.cfg.width,
            "height": self.cfg.height,
            "fov": self.cfg.fov,
            "elevation": elevs,
            "azimuth": azims,
        }


class RandomMVDreamCameraDataset(InfiniteDataset):
    @ignore_kwargs
    @dataclass
    class Config:
        width: int = 512
        height: int = 512
        dist_range: Tuple[float, float] = (1.8, 2.2)
        elev_range: Tuple[float, float] = (-5, 40)
        azim_range: Tuple[float, float] = (0, 90)
        fov: float = 72
        up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z"
        convention: Literal[
            "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
        ] = "RDF"
        batch_size: int = 1
        device: str = "cuda"

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = self.Config(**cfg)
        assert self.cfg.batch_size % 4 == 0, "Batch size must be a multiple of 4"

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        c2ws, Ks, elevs, azims = [], [], [], []
        for _ in range(self.cfg.batch_size // 4):
            dist = np.random.uniform(*self.cfg.dist_range)
            elev = np.random.uniform(*self.cfg.elev_range)
            azim1 = np.random.uniform(*self.cfg.azim_range)
            azim2 = (azim1 + 90) % 360
            azim3 = (azim1 + 180) % 360
            azim4 = (azim1 + 270) % 360
            for azim in [azim1, azim2, azim3, azim4]:
                c2w, K = generate_camera_params(
                    dist,
                    elev,
                    azim,
                    self.cfg.fov,
                    self.cfg.width,
                    self.cfg.height,
                    self.cfg.up_vec,
                    self.cfg.convention,
                )
                c2w = c2w.to(self.cfg.device)
                K = K.to(self.cfg.device)
                c2ws.append(c2w)
                Ks.append(K)
                elevs.append(elev)
                azims.append(azim)

        c2ws = torch.stack(c2ws)
        Ks = torch.stack(Ks)

        return {
            "batch_size": self.cfg.batch_size,
            "c2w": c2ws,
            "K": Ks,
            "width": self.cfg.width,
            "height": self.cfg.height,
            "fov": self.cfg.fov,
            "elevation": elevs,
            "azimuth": azims,
        }


class SeqTurnaroundCameraDataset(Dataset):
    @ignore_kwargs
    @dataclass
    class Config:
        num_cameras: int = 60
        dist: float = 2.0
        elev: float = 30.0
        fov: float = 72
        width: int = 512
        height: int = 512
        up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z"
        convention: Literal[
            "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
        ] = "RDF"
        batch_size: int = 1
        device: str = "cuda"

    def __init__(self, cfg) -> None:
        self.cfg = self.Config(**cfg)
        self.azimuths = np.linspace(0, 360, self.cfg.num_cameras, endpoint=False)
        self.count = 0

    def __len__(self) -> int:
        return self.cfg.num_cameras

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        azim = self.azimuths[idx]
        c2w, K = generate_camera_params(
            self.cfg.dist,
            self.cfg.elev,
            azim,
            self.cfg.fov,
            self.cfg.width,
            self.cfg.height,
            self.cfg.up_vec,
            self.cfg.convention,
        )
        c2w = c2w.to(self.cfg.device)
        K = K.to(self.cfg.device)

        return {
            "batch_size": self.cfg.batch_size,
            "c2w": c2w.unsqueeze(0),
            "K": K.unsqueeze(0),
            "width": self.cfg.width,
            "height": self.cfg.height,
            "fov": self.cfg.fov,
            "azimuth": [azim],
            "elevation": [self.cfg.elev],
        }

    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.__getitem__(self.count)
        self.count += 1
        if self.count == len(self):
            self.count = 0
        return data
    

# from model.nvdiff_render.mesh import *
# from model.nvdiff_render.render import *
# from model.nvdiff_render.texture import *
# from model.nvdiff_render.material import *
# from model.nvdiff_render.obj import *
# class NVDiffrastCameraDataset(InfiniteDataset):
#     @ignore_kwargs
#     @dataclass
#     class Config:
#         width: int = 512
#         height: int = 512
#         dist_range: Tuple[float, float] = (1.8, 2.2)
#         elev_range: Tuple[float, float] = (-5, 40)
#         azim_range: Tuple[float, float] = (0, 360)
#         fov: float = 72
#         up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z"
#         convention: Literal[
#             "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
#         ] = "RDF"
#         batch_size: int = 1
#         device: str = "cuda"

#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = self.Config(**cfg)

#     def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         spp = 1
#         cam_near_far=[0.1, 1000.0]
#         iter_res = [self.cfg.height, self.cfg.width]
#         fovy = np.deg2rad(45)
#         proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

#         # Random rotation/translation matrix for optimization.
#         mv_list, mvp_list, campos_list, direction_list = [], [], [], []
#         for view_i in range(self.cfg.batch_size):
#             cam_radius = np.random.uniform(self.cfg.dist_range[0], self.cfg.dist_range[1])
#             angle_x = np.random.uniform(self.cfg.elev_range[0] * np.pi / 180, self.cfg.elev_range[1] * np.pi / 180)
#             angle_y = np.random.uniform(self.cfg.azim_range[0] * np.pi / 180, self.cfg.azim_range[1] * np.pi / 180)

#             # direction
#             # 0 = front, 1 = side, 2 = back, 3 = overhead
#             if angle_x < -np.pi / 4:
#                 direction = 3
#             else:
#                 if 0 <= angle_y <= np.pi / 4 or angle_y > 7 * np.pi / 4:
#                     direction = 0
#                 elif np.pi / 4 < angle_y <= 3 * np.pi / 4:
#                     direction = 1
#                 elif 3 * np.pi / 4 < angle_y <= 5 * np.pi / 4:
#                     direction = 2
#                 elif 5 * np.pi / 4 < angle_y <= 7 * np.pi / 4:
#                     direction = 1

#             # for object, hard to tell front, back. so, perform prompt augment for only overhead view
#             # If the results do not look good, you may use this direction prompts.
#             # if angle_x < -np.pi / 4:
#             #     direction = 1
#             # else:
#             #     direction = 0

#             mv = util.translate(0, 0, -cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
#             mvp = proj_mtx @ mv
#             campos = torch.linalg.inv(mv)[:3, 3]
#             mv_list.append(mv[None, ...].cuda())
#             mvp_list.append(mvp[None, ...].cuda())
#             campos_list.append(campos[None, ...].cuda())
#             direction_list.append(direction)

#         cam = {
#             'mv': torch.cat(mv_list, dim=0),
#             'mvp': torch.cat(mvp_list, dim=0),
#             'campos': torch.cat(campos_list, dim=0),
#             'direction': np.array(direction_list, dtype=np.int32),
#             'resolution': iter_res,
#             'spp': spp,
#             'batch_size': self.cfg.batch_size
#         }
#         return cam

# class NVDiffrastMVDreamCameraDataset(InfiniteDataset):
#     @ignore_kwargs
#     @dataclass
#     class Config:
#         width: int = 512
#         height: int = 512
#         dist_range: Tuple[float, float] = (1.8, 2.2)
#         elev_range: Tuple[float, float] = (-5, 40)
#         azim_range: Tuple[float, float] = (0, 360)
#         fov: float = 72
#         up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z"
#         convention: Literal[
#             "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
#         ] = "RDF"
#         batch_size: int = 1
#         device: str = "cuda"

#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = self.Config(**cfg)
#         assert self.cfg.batch_size % 4 == 0, "Batch size must be a multiple of 4"

#     def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         spp = 1
#         cam_near_far=[0.1, 1000.0]
#         iter_res = [self.cfg.height, self.cfg.width]
#         fovy = np.deg2rad(45)
#         proj_mtx = util.perspective(fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])
        
#         elevs = np.random.uniform(*self.cfg.elev_range, self.cfg.batch_size // 4)
#         azims = np.random.uniform(*self.cfg.azim_range, self.cfg.batch_size // 4)

#         elevs = np.concatenate([elevs, elevs, elevs, elevs])
#         azims = np.concatenate([azims, azims + 90, azims + 180, azims + 270])

#         # Random rotation/translation matrix for optimization.
#         mv_list, mvp_list, campos_list, direction_list = [], [], [], []
#         for view_i in range(self.cfg.batch_size):
#             cam_radius = np.random.uniform(self.cfg.dist_range[0], self.cfg.dist_range[1])
#             #angle_x = np.random.uniform(self.cfg.elev_range[0] * np.pi / 180, self.cfg.elev_range[1] * np.pi / 180)
#             #angle_y = np.random.uniform(self.cfg.azim_range[0] * np.pi / 180, self.cfg.azim_range[1] * np.pi / 180)
#             angle_x = elevs[view_i] * np.pi / 180
#             angle_y = azims[view_i] * np.pi / 180

#             # direction
#             # 0 = front, 1 = side, 2 = back, 3 = overhead
#             if angle_x < -np.pi / 4:
#                 direction = 3
#             else:
#                 if 0 <= angle_y <= np.pi / 4 or angle_y > 7 * np.pi / 4:
#                     direction = 0
#                 elif np.pi / 4 < angle_y <= 3 * np.pi / 4:
#                     direction = 1
#                 elif 3 * np.pi / 4 < angle_y <= 5 * np.pi / 4:
#                     direction = 2
#                 elif 5 * np.pi / 4 < angle_y <= 7 * np.pi / 4:
#                     direction = 1

#             # for object, hard to tell front, back. so, perform prompt augment for only overhead view
#             # If the results do not look good, you may use this direction prompts.
#             # if angle_x < -np.pi / 4:
#             #     direction = 1
#             # else:
#             #     direction = 0

#             mv = util.translate(0, 0, -cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
#             mvp = proj_mtx @ mv
#             campos = torch.linalg.inv(mv)[:3, 3]
#             mv_list.append(mv[None, ...].cuda())
#             mvp_list.append(mvp[None, ...].cuda())
#             campos_list.append(campos[None, ...].cuda())
#             direction_list.append(direction)

#         cam = {
#             'mv': torch.cat(mv_list, dim=0),
#             'mvp': torch.cat(mvp_list, dim=0),
#             'campos': torch.cat(campos_list, dim=0),
#             'direction': np.array(direction_list, dtype=np.int32),
#             'resolution': iter_res,
#             'spp': spp,
#             'batch_size': self.cfg.batch_size,
#             'elevation': elevs,
#             'azimuth': azims
#         }
#         return cam