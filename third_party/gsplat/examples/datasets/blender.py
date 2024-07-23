import os
from math import atan, tan
from typing import Any, Dict, List, Optional
import json

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

def fov_to_focal_length(fov, hole_rad=0.5):
    return hole_rad / tan(fov / 2)


def focal_length_to_fov(focal_length, hole_rad=0.5):
    return 2 * atan(hole_rad / focal_length)

def get_intrinsics(fov, height, width, invert_y=False):
    focal_length = fov_to_focal_length(fov)
    fx = focal_length * width
    fy = focal_length * height
    cx = width / 2.0
    cy = height / 2.0

    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    if invert_y:
        intrinsics[1, 1] *= -1

    return intrinsics

CAM_TRANSFORMATIONS = {
    "RUF": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "Unity": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),

    "Pytorch3D": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "LUF": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),

    "OpenGL": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    "RUB": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),

    "OpenCV": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    "RDF": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
}

def convert_camera_convention(C2Ws, from_convention, to_convention):
    # C2Ws: B x 4 x 4
    # from_convention, to_convention: str
    # returns: B x 4 x 4

    from_ruf = CAM_TRANSFORMATIONS[from_convention]
    ruf_to = CAM_TRANSFORMATIONS[to_convention]
    from_to = np.linalg.inv(from_ruf) @ ruf_to
    new_C2Ws = C2Ws.copy()
    new_C2Ws[:, :3, :3] = new_C2Ws[:, :3, :3] @ from_to
    return new_C2Ws

def load_blender_cameras(path, target_size=[512, 512]):
    train_cam_path = os.path.join(path, "transforms_test.json")
    target_height, target_width = target_size

    C2Ws = []
    Ks = []
    img_names = []
    img_paths = []

    with open(train_cam_path) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        extension=".png"

        for frame in frames:
            c2w = np.array(frame["transform_matrix"])
            FovY = focal_length_to_fov(fov_to_focal_length(fovx, target_width/2), target_height/2)
            k = get_intrinsics(FovY, target_height, target_width, invert_y=False)


            cam_name = frame["file_path"] + extension
            image_name = frame["file_path"]
            image_path = os.path.join(path, cam_name)

            C2Ws.append(c2w)
            Ks.append(k)
            img_names.append(image_name)
            img_paths.append(image_path)
    C2Ws = np.stack(C2Ws, axis=0)
    C2Ws = convert_camera_convention(C2Ws, "OpenGL", "OpenCV")
    C2Ws = [C2Ws[i] for i in range(C2Ws.shape[0])]
            
    return C2Ws, Ks, img_names, img_paths

class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 4,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.test_every = test_every
        
        c2w_mats, Ks, image_names, image_paths = load_blender_cameras(data_dir, target_size=[800, 800])

        num_cams = len(c2w_mats)
        camera_ids = list(range(num_cams))
        Ks_dict = {i: Ks[i] for i in range(num_cams)}
        params_dict = {i: np.empty(0, dtype=np.float32) for i in range(num_cams)}
        imsize_dict = {i: (800, 800) for i in range(num_cams)}

        print(
            f"[Parser] {len(camera_ids)} images."
        )

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.stack(c2w_mats, axis=0)

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        # inds = np.argsort(image_names)
        # image_names = [image_names[i] for i in inds]
        # camtoworlds = camtoworlds[inds]
        # camera_ids = [camera_ids[i] for i in inds]

        # 3D points and {image_name -> [point_idx]}
        points = None
        points_err = None
        points_rgb = None
        point_indices = dict()

        # Normalize the world space.
        transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            #self.indices = indices[indices % self.parser.test_every != 0]
            self.indices = indices
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])# [..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image[...,:3]).float(),
            "alpha": torch.from_numpy(image[...,3:]).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            if self.patch_size is not None:
                points[:, 0] -= x
                points[:, 1] -= y
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
