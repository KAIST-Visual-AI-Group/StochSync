import math
import torch
import torch.nn.functional as F
from utils.extra_utils import accumulate_tensor
from utils.print_utils import print_warning


def normalize_grid(grid, height, width):
    """
    Normalize map coordinates to the range [-1, 1] for use with grid_sample.
    """
    grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0
    return grid


def remap(image, grid, mode="bilinear"):
    """
    Remap an image based on provided coordinate maps using grid_sample.
    """
    B, C, H, W = image.shape
    grid = normalize_grid(grid, H, W).unsqueeze(0)
    remapped_image = F.grid_sample(
        image, grid, mode=mode, padding_mode="border", align_corners=True
    )
    return remapped_image


def remap_int(tensor, grid, indexing="xy"):
    H, W = tensor.shape[-2:]
    if indexing == "xy":
        grid = grid.flip(-1)
    h_idx = grid[:, :, 0].long()
    w_idx = grid[:, :, 1].long()
    h_idx = h_idx.clamp(0, H - 1)
    w_idx = w_idx.clamp(0, W - 1)
    return tensor[..., h_idx, w_idx]


def xyz_to_lonlat(xyz):
    """
    Convert XYZ coordinates to longitude and latitude.
    """
    norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    xyz_norm = xyz / norm
    x, y, z = xyz_norm[..., 0], xyz_norm[..., 1], xyz_norm[..., 2]

    lon = torch.atan2(x, z)
    lat = torch.asin(y)

    return torch.stack([lon, lat], dim=-1)


def lonlat_to_xyz(lonlat):
    """
    Convert longitude and latitude to XYZ coordinates.
    """
    lon, lat = lonlat[..., 0], lonlat[..., 1]

    x = torch.cos(lat) * torch.sin(lon)
    y = torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)

    return torch.stack([x, y, z], dim=-1)


def project_xyz(xyz, K, eps=1e-6):
    """
    Project XYZ coordinates to the image plane using the intrinsic matrix K.
    """
    valid_mask = xyz[..., 2] > eps
    xyz_proj = xyz.clone()
    xyz_proj[~valid_mask] = float("inf")
    xyz_proj[valid_mask] /= xyz_proj[valid_mask][..., 2:3]
    xyz_proj[valid_mask] = torch.matmul(xyz_proj[valid_mask], K.T)
    return xyz_proj[..., :2]


def compute_intrinsic_matrix(fov, width, height):
    """
    Compute the camera intrinsic matrix based on FOV and image dimensions.
    """
    f = 0.5 * height / math.tan(0.5 * fov * math.pi / 180)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=torch.float32)


def rodriques(v):
    """
    Compute the Rodrigues rotation matrix.
    """
    theta = torch.linalg.norm(v)
    if theta == 0:
        return torch.eye(3)
    v = v / theta
    K = torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=torch.float32
    )
    return torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K


def rotation_matrix(theta, phi):
    """
    Compute the rotation matrix based on theta and phi angles.
    """
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    R1 = rodriques(y_axis * math.radians(theta))
    R2 = rodriques(torch.matmul(R1, x_axis) * math.radians(phi))

    return R2 @ R1


def lonlat_to_xy(lonlat, height, width):
    """
    Convert longitude and latitude to pixel coordinates.
    """
    X = (lonlat[..., 0] / (2 * math.pi) + 0.5) * width
    Y = (lonlat[..., 1] / math.pi + 0.5) * height
    return torch.stack([X, Y], dim=-1)


def xy_to_lonlat(xy, height, width):
    """
    Convert pixel coordinates to longitude and latitude.
    """
    lon = (xy[..., 0] / width - 0.5) * 2 * math.pi
    lat = (xy[..., 1] / height - 0.5) * math.pi
    return torch.stack([lon, lat], dim=-1)


def compute_pano2pers_map(
    fov, theta, phi, pers_height, pers_width, pano_height, pano_width
):
    """
    Compute the XY mapping from the panorama to perspective view.
    """
    K = compute_intrinsic_matrix(fov, pers_width, pers_height)
    K_inv = torch.inverse(K)

    x, y = torch.meshgrid(
        torch.arange(pers_width), torch.arange(pers_height), indexing="xy"
    )
    xyz = torch.stack([x.float(), y.float(), torch.ones_like(x).float()], dim=-1)
    xyz = torch.matmul(xyz, K_inv.T)

    R = rotation_matrix(theta, phi)
    xyz = torch.matmul(xyz, R.T)

    lonlat = xyz_to_lonlat(xyz)
    return lonlat_to_xy(lonlat, pano_height, pano_width)


def compute_pers2pano_map(
    fov, theta, phi, pers_height, pers_width, pano_height, pano_width
):
    """
    Compute the inverse XY mapping from the perspective view to panorama.
    """
    K = compute_intrinsic_matrix(fov, pers_width, pers_height)

    x, y = torch.meshgrid(
        torch.arange(pano_width), torch.arange(pano_height), indexing="xy"
    )
    lonlat = xy_to_lonlat(torch.stack([x, y], dim=-1), pano_height, pano_width)

    R = rotation_matrix(theta, phi)
    xyz = lonlat_to_xyz(lonlat)
    xyz = torch.matmul(xyz, R)

    return project_xyz(xyz, K)


def pano_to_pers(panorama, pano2pers, mode="nearest"):
    perspective_image = remap(panorama, pano2pers, mode)
    return perspective_image


def pano_to_pers_raw(panorama, fov, theta, phi, pers_height, pers_width, mode="nearest"):
    """
    Transform a panorama image to a perspective view.
    """
    pano_height, pano_width = panorama.shape[-2], panorama.shape[-1]
    pano2pers = compute_pano2pers_map(
        fov, theta, phi, pers_height, pers_width, pano_height, pano_width
    )
    pano2pers = pano2pers.to(panorama.device)
    return pano_to_pers(panorama, pano2pers, mode)


def pano_to_pers_accum(panorama, pers2pano, pers_height, pers_width):
    """
    Transform a panorama image to a perspective view.
    """
    H, W = pers_height, pers_width

    idx_pers = torch.arange(H * W).view(H, W).to(panorama.device)
    idx_pano, mask_pano = pers_to_pano(idx_pers, pers2pano, return_mask=True)
    idx_flat = idx_pano[mask_pano]
    val_flat = panorama[..., mask_pano]
    pers_shape = (*panorama.shape[:-2], H, W)
    pers_flat_shape = (*panorama.shape[:-2], H * W)
    pers, cnt = accumulate_tensor(
        torch.zeros(pers_flat_shape, device=panorama.device), idx_flat, val_flat
    )
    pers = pers.view(*pers_shape)
    cnt = cnt.view(*pers_shape)
    return pers, cnt


def pano_to_pers_accum_raw(panorama, fov, theta, phi, pers_height, pers_width):
    """
    Transform a panorama image to a perspective view.
    """
    pano_height, pano_width = panorama.shape[-2], panorama.shape[-1]
    pers2pano = compute_pers2pano_map(
        fov, theta, phi, pers_height, pers_width, pano_height, pano_width
    )
    pers2pano = pers2pano.to(panorama.device)
    return pano_to_pers_accum(panorama, pers2pano, pers_height, pers_width)


def pers_to_pano(perspective, pers2pano, return_mask=False, mode="nearest"):
    pers_height, pers_width = perspective.shape[-2], perspective.shape[-1]
    safe_padding = 1.0
    valid_mask = (
        (pers2pano[..., 0] >= -0.5 - safe_padding)
        & (pers2pano[..., 0] < pers_width - 0.5 + safe_padding)
        & (pers2pano[..., 1] >= -0.5 - safe_padding)
        & (pers2pano[..., 1] < pers_height - 0.5 + safe_padding)
    )

    if perspective.dtype == torch.float32:
        panorama_image = remap(perspective, pers2pano, mode) * valid_mask
    else:
        panorama_image = remap_int(perspective, pers2pano.round().long()) * valid_mask

    if return_mask:
        return panorama_image, valid_mask
    return panorama_image


def pers_to_pano_raw(
    perspective, fov, theta, phi, pano_height, pano_width, return_mask=False, mode="nearest"
):
    """
    Transform a perspective view image to a panorama.
    """
    pers_height, pers_width = perspective.shape[-2], perspective.shape[-1]
    pers2pano = compute_pers2pano_map(
        fov, theta, phi, pers_height, pers_width, pano_height, pano_width
    )
    pers2pano = pers2pano.to(perspective.device)
    return pers_to_pano(perspective, pers2pano, return_mask, mode)
