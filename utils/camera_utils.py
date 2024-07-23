from math import pi, tan, atan
from typing import Literal

import numpy as np
import torch


def fov_to_focal_length(fov):
    return 0.5 / tan(fov / 2)


def focal_length_to_fov(focal_length, hole_rad=0.5):
    return 2 * atan(hole_rad / focal_length)


def dfov_to_focal_length(dfov):
    fov = dfov * pi / 180
    return 0.5 / tan(fov / 2)


def focal_length_to_dfov(focal_length):
    fov = 2 * atan(0.5 / focal_length)
    return fov * 180 / pi


def get_intrinsics(fov, height, width, invert_y=False):
    """
    Generates the camera intrinsic matrix based on field of view (fov), image height, and width.

    Args:
        fov (float): Field of view in radians.
        height (int): Height of the image in pixels.
        width (int): Width of the image in pixels.

    Returns:
        torch.Tensor: The camera intrinsic matrix.
    """
    focal_length = fov_to_focal_length(fov)
    fx = focal_length * width
    fy = focal_length * height
    cx = width / 2.0
    cy = height / 2.0

    intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    if invert_y:
        intrinsics[1, 1] *= -1

    return intrinsics


def get_window_matrix(width, height):
    # -1 ~ 1 to 0 ~ width, 0 ~ height
    window_matrix = torch.tensor(
        [[width / 2, 0, width / 2], [0, -height / 2, height / 2], [0, 0, 1]]
    )

    return window_matrix


def get_projection_matrix(fov, aspect_ratio, near, far):
    """
    Generates a perspective projection matrix.

    Args:
        fov (float): Field of view in radians.
        aspect_ratio (float): Aspect ratio of the viewport (width / height).
        near (float): The near clipping plane.
        far (float): The far clipping plane.

    Returns:
        torch.Tensor: The projection matrix.
    """
    focal_length = fov_to_focal_length(fov)
    f = focal_length
    range_inv = 1.0 / (near - far)

    proj_matrix = torch.tensor(
        [
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (near + far) * range_inv, 2 * near * far * range_inv],
            [0, 0, -1, 0],
        ]
    )

    return proj_matrix


def get_rotation_matrix(eye, target, up):
    """
    Generates a rotation matrix based on the camera position, target position, and up direction.

    Args:
        eye (torch.Tensor): The camera position.
        target (torch.Tensor): The target position where the camera is looking.
        up (torch.Tensor): The up direction vector.

    Returns:
        torch.Tensor: The rotation matrix.
    """
    zaxis = torch.nn.functional.normalize(eye - target, dim=0)
    xaxis = torch.nn.functional.normalize(torch.cross(up, zaxis), dim=0)
    yaxis = torch.cross(zaxis, xaxis)

    rotation_matrix = torch.stack([xaxis, yaxis, zaxis], dim=1)

    assert torch.abs(torch.det(rotation_matrix) - 1) < 1e-6, rotation_matrix

    return rotation_matrix


def get_inverse_view_matrix(eye, target, up):
    """
    Generates an inverse view matrix (camera-to-world matrix).

    Args:
        eye (torch.Tensor): The camera position.
        target (torch.Tensor): The target position where the camera is looking.
        up (torch.Tensor): The up direction vector.

    Returns:
        torch.Tensor: The inverse view matrix.
    """
    zaxis = torch.nn.functional.normalize(eye - target, dim=0)
    xaxis = torch.nn.functional.normalize(torch.cross(up, zaxis), dim=0)
    yaxis = torch.cross(zaxis, xaxis)

    inv_view_matrix = torch.eye(4)
    inv_view_matrix[:3, 0] = xaxis
    inv_view_matrix[:3, 1] = yaxis
    inv_view_matrix[:3, 2] = zaxis
    inv_view_matrix[:3, 3] = eye

    # it is invertible (close to 0) eps = 1e-6
    assert torch.abs(torch.det(inv_view_matrix)) > 1e-6, inv_view_matrix

    return inv_view_matrix


def get_view_matrix(eye, target, up):
    """
    Generates a view matrix (world-to-camera matrix).

    Args:
        eye (torch.Tensor): The camera position.
        target (torch.Tensor): The target position where the camera is looking.
        up (torch.Tensor): The up direction vector.

    Returns:
        torch.Tensor: The view matrix.
    """
    return torch.inverse(get_inverse_view_matrix(eye, target, up))


def get_c2w_matrix(eye, target, up):
    return get_inverse_view_matrix(eye, target, up)


def get_w2c_matrix(eye, target, up):
    return get_view_matrix(eye, target, up)


def get_model_view_projection_matrix(model_matrix, view_matrix, projection_matrix):
    """
    Generates the Model-View-Projection (MVP) matrix.

    Args:
        model_matrix (torch.Tensor): The model matrix.
        view_matrix (torch.Tensor): The view matrix.
        projection_matrix (torch.Tensor): The projection matrix.

    Returns:
        torch.Tensor: The Model-View-Projection matrix.
    """
    return projection_matrix @ view_matrix @ model_matrix


def generate_camera_params(
    dist,
    elev,
    azim,
    fov,
    width,
    height,
    up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z",
    convention: Literal[
        "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
    ] = "RDF",
):
    elev = elev * np.pi / 180
    azim = azim * np.pi / 180
    fov = fov * np.pi / 180

    world_up, world_front, world_right = {
        "x": (np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, 1, 0])),
        "y": (np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])),
        "z": (np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0])),
        "-x": (np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
        "-y": (np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])),
        "-z": (np.array([0, 0, -1]), np.array([1, 0, 0]), np.array([0, 1, 0])),
    }[up_vec]

    cam_pos = dist * (
        world_up * np.sin(elev)
        - world_front * np.cos(elev) * np.cos(azim)
        + world_right * np.cos(elev) * np.sin(azim)
    )

    lookat = -cam_pos / np.linalg.norm(cam_pos)
    fake_up = world_up

    right = np.cross(lookat, fake_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, lookat)

    invert_y = False
    if convention == "LUF" or convention == "Pytorch3D":
        R = np.stack([-right, up, lookat], axis=1)
        invert_y = True
    if convention == "RDF" or convention == "OpenCV":
        R = np.stack([right, -up, lookat], axis=1)
        invert_y = False
    elif convention == "RUB" or convention == "OpenGL":
        R = np.stack([right, up, -lookat], axis=1)
        invert_y = True
    elif convention == "RUF" or convention == "Unity":
        R = np.stack([right, up, lookat], axis=1)
        invert_y = True

    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam_pos

    K = get_intrinsics(fov, height, width, invert_y=invert_y)
    return torch.tensor(c2w, dtype=torch.float32), K


def generate_camera(
    dist,
    elev,
    azim,
    fov,
    width,
    height,
    up_vec: Literal["x", "y", "z", "-x", "-y", "-z"] = "z",
    convention: Literal[
        "LUF", "RDF", "RUB", "RUF", "Pytorch3D", "OpenCV", "OpenGL", "Unity"
    ] = "RDF",
    device="cpu",
):
    c2w, K = generate_camera_params(
        dist,
        elev,
        azim,
        fov,
        width,
        height,
        up_vec=up_vec,
        convention=convention,
    )
    c2w = torch.tensor(c2w, dtype=torch.float32).to(device)
    K = torch.tensor(K, dtype=torch.float32).to(device)

    return {
        "c2w": c2w.unsqueeze(0),
        "K": K.unsqueeze(0),
        "width": width,
        "height": height,
        "azimuth": [azim],
        "elevation": [elev],
    }

def merge_camera(cam_list):
    c2w = torch.cat([cam["c2w"] for cam in cam_list], dim=0)
    K = torch.cat([cam["K"] for cam in cam_list], dim=0)
    width = cam_list[0]["width"]
    height = cam_list[0]["height"]
    azimuth = [azim for cam in cam_list for azim in cam["azimuth"]]
    elevation = [elev for cam in cam_list for elev in cam["elevation"]]

    return {
        "c2w": c2w,
        "K": K,
        "width": width,
        "height": height,
        "azimuth": azimuth,
        "elevation": elevation,
    }