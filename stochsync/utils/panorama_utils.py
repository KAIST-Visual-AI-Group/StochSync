import math
import torch
import torch.nn.functional as F
from functools import lru_cache
from .extra_utils import accumulate_tensor
from .matrix_utils import quat_to_rot


def normalize_grid(grid, height, width):
    """
    Normalize map coordinates to the range [-1, 1] for use with grid_sample.
    """
    grid = grid.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / (width - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (height - 1) - 1.0
    return grid


def gather_2d(tensor, grid):
    # tensor: BCHW -> BCX
    # grid: Bhw2 -> BHW -> B1
    # output: BChw
    B, C, H, W = tensor.shape
    b, h, w, _ = grid.shape
    assert b == B, f"Batch size mismatch: {b} != {B}"

    tensor = tensor.contiguous()
    lin_idx = (
        (grid[..., 0] * tensor.shape[-1] + grid[..., 1])
        .view(B, 1, -1)
        .expand(-1, C, -1)
    )
    tensor = tensor.view(B, C, tensor.shape[-2] * tensor.shape[-1])
    return tensor.gather(-1, lin_idx).view(B, C, h, w)


def remap(image, grid, mode="bilinear", padding_mode="border"):
    """
    Remap an image based on provided coordinate maps using grid_sample.
    """
    B, C, H, W = image.shape
    grid = normalize_grid(grid, H, W)
    grid = grid.unsqueeze(0) if grid.dim() == 3 else grid
    remapped_image = F.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )
    return remapped_image


def remap_int(tensor, grid, indexing="xy"):
    H, W = tensor.shape[-2:]
    if indexing == "xy":
        grid = grid.flip(-1)

    grid = grid.unsqueeze(0) if grid.dim() == 3 else grid
    grid = grid.long()
    grid[..., 0] = grid[..., 0].clamp(0, H - 2)
    grid[..., 1] = grid[..., 1].clamp(0, W - 2)
    return gather_2d(tensor, grid)


def remap_max(tensor, grid, indexing="xy"):
    """
    Remap an image based on provided coordinate maps using max pooling between neighbors.
    """
    H, W = tensor.shape[-2:]
    if indexing == "xy":
        grid = grid.flip(-1)

    grid = grid.unsqueeze(0) if grid.dim() == 3 else grid
    grid = grid.long()
    grid[..., 0] = grid[..., 0].clamp(0, H - 2)
    grid[..., 1] = grid[..., 1].clamp(0, W - 2)

    tensor1 = gather_2d(tensor, grid)
    tensor2 = gather_2d(tensor, grid + torch.tensor([0, 1], device=grid.device))
    tensor3 = gather_2d(tensor, grid + torch.tensor([1, 0], device=grid.device))
    tensor4 = gather_2d(tensor, grid + torch.tensor([1, 1], device=grid.device))
    tensors = torch.stack([tensor1, tensor2, tensor3, tensor4], dim=-1)
    return torch.max(tensors, dim=-1).values


def remap_min(tensor, grid, indexing="xy"):
    """
    Remap an image based on provided coordinate maps using min pooling between neighbors.
    """
    H, W = tensor.shape[-2:]
    if indexing == "xy":
        grid = grid.flip(-1)

    grid = grid.unsqueeze(0) if grid.dim() == 3 else grid
    grid = grid.long()
    grid[..., 0] = grid[..., 0].clamp(0, H - 2)
    grid[..., 1] = grid[..., 1].clamp(0, W - 2)

    tensor1 = gather_2d(tensor, grid)
    tensor2 = gather_2d(tensor, grid + torch.tensor([0, 1], device=grid.device))
    tensor3 = gather_2d(tensor, grid + torch.tensor([1, 0], device=grid.device))
    tensor4 = gather_2d(tensor, grid + torch.tensor([1, 1], device=grid.device))
    tensors = torch.stack([tensor1, tensor2, tensor3, tensor4], dim=-1)
    return torch.min(tensors, dim=-1).values


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


def rotation_matrix(theta, phi, roll=0):
    """
    Compute the rotation matrix based on theta and phi angles.
    """
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    R0 = rodriques(z_axis * math.radians(roll))
    R1 = rodriques(x_axis * math.radians(phi))
    R2 = rodriques(y_axis * math.radians(theta))

    return R2 @ R1 @ R0


def lonlat_to_xy(lonlat, height, width, lonrange=(-math.pi, math.pi), latrange=(-math.pi / 2, math.pi / 2)):
    """
    Convert longitude and latitude to pixel coordinates.
    """
    nlon = (lonlat[..., 0] - lonrange[0]) / (lonrange[1] - lonrange[0])
    nlat = (lonlat[..., 1] - latrange[0]) / (latrange[1] - latrange[0])
    X = nlon * width
    Y = nlat * height
    return torch.stack([X, Y], dim=-1)

def xy_to_lonlat(xy, height, width, lonrange=(-math.pi, math.pi), latrange=(-math.pi / 2, math.pi / 2)):
    """
    Convert pixel coordinates to longitude and latitude.
    """
    nx = xy[..., 0] / width
    ny = xy[..., 1] / height
    lon = nx * (lonrange[1] - lonrange[0]) + lonrange[0]
    lat = ny * (latrange[1] - latrange[0]) + latrange[0]
    return torch.stack([lon, lat], dim=-1)


def lonlat_to_xy_plane(lonlat, height, width, center_lat=90, edge_lat=-36):
    """
    Convert longitude and latitude to pixel coordinates with plane projection.
    """
    r = (lonlat[..., 1] * 180 / math.pi - center_lat) / (edge_lat - center_lat)
    X = width * (r * torch.cos(lonlat[..., 0]) + 1) / 2
    Y = height * (r * torch.sin(lonlat[..., 0]) + 1) / 2
    return torch.stack([X, Y], dim=-1)


def xy_to_lonlat_plane(xy, height, width, center_lat=90, edge_lat=-36):
    """
    Convert pixel coordinates to longitude and latitude with plane projection.
    """
    # lon = (xy[..., 0] / width - 0.5) * 2 * math.pi
    # lat = (xy[..., 1] / height - 0.5) * math.pi
    xy = 2 * xy / torch.tensor([width, height], dtype=xy.dtype, device=xy.device) - 1
    lon = torch.atan2(xy[..., 1], xy[..., 0])
    lat = torch.linalg.norm(xy, dim=-1) * (edge_lat - center_lat) + center_lat
    lat = lat * math.pi / 180
    return torch.stack([lon, lat], dim=-1)

@lru_cache(maxsize=32)
def compute_pano2pers_map(
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    pano_height,
    pano_width,
    quat=None,
    lonlat_to_xy=lonlat_to_xy,
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

    R_base = torch.eye(3)
    if quat is not None:
        R_base = quat_to_rot(quat)
    
    R = R_base @ rotation_matrix(theta, phi)
    xyz = torch.matmul(xyz, R.T)

    lonlat = xyz_to_lonlat(xyz)
    return lonlat_to_xy(lonlat, pano_height, pano_width)


@lru_cache(maxsize=32)
def compute_pers2pano_map(
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    pano_height,
    pano_width,
    quat=None,
    xy_to_lonlat=xy_to_lonlat,
):
    """
    Compute the inverse XY mapping from the perspective view to panorama.
    """
    K = compute_intrinsic_matrix(fov, pers_width, pers_height)

    x, y = torch.meshgrid(
        torch.arange(pano_width), torch.arange(pano_height), indexing="xy"
    )
    lonlat = xy_to_lonlat(torch.stack([x, y], dim=-1), pano_height, pano_width)

    R_base = torch.eye(3)
    if quat is not None:
        R_base = quat_to_rot(quat)
    
    R = R_base @ rotation_matrix(theta, phi)
    xyz = lonlat_to_xyz(lonlat)
    xyz = torch.matmul(xyz, R)

    return project_xyz(xyz, K)


def compute_sp2pers_map(
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    pano_height,
    pano_width,
    quat=None,
    lonlat_to_xy=lonlat_to_xy,
):
    """
    Compute the XY mapping from the panorama to perspective view.
    """

    a, b = torch.meshgrid(
        torch.linspace(-math.radians(fov / 2), math.radians(fov / 2), pers_width),
        torch.linspace(-math.radians(fov / 2), math.radians(fov / 2), pers_height),
        indexing="xy",
    )
    tmp = 1 / torch.sqrt(1 - torch.sin(a) ** 2 * torch.sin(b) ** 2)
    x = tmp * torch.sin(a) * torch.cos(b)
    y = tmp * torch.cos(a) * torch.sin(b)
    z = tmp * torch.cos(a) * torch.cos(b)
    xyz = torch.stack([x, y, z], dim=-1)

    R_base = torch.eye(3)
    if quat is not None:
        R_base = quat_to_rot(quat)

    R = R_base @ rotation_matrix(theta, phi)
    xyz = torch.matmul(xyz, R.T)

    lonlat = xyz_to_lonlat(xyz)
    return lonlat_to_xy(lonlat, pano_height, pano_width)


def compute_pers2sp_map(
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    pano_height,
    pano_width,
    quat=None,
    xy_to_lonlat=xy_to_lonlat,
):
    """
    Compute the inverse XY mapping from the perspective view to panorama.
    """
    x, y = torch.meshgrid(
        torch.arange(pano_width), torch.arange(pano_height), indexing="xy"
    )
    lonlat = xy_to_lonlat(torch.stack([x, y], dim=-1), pano_height, pano_width)

    R_base = torch.eye(3)
    if quat is not None:
        R_base = quat_to_rot(quat)

    R = R_base @ rotation_matrix(theta, phi)
    xyz = lonlat_to_xyz(lonlat)
    xyz = torch.matmul(xyz, R)

    a = torch.atan2(xyz[..., 0], xyz[..., 2])
    a = (a + math.radians(fov) / 2) / math.radians(fov) * pers_width
    b = torch.atan2(xyz[..., 1], xyz[..., 2])
    b = (b + math.radians(fov) / 2) / math.radians(fov) * pers_height

    return torch.stack([a, b], dim=-1)
    # return project_xyz(xyz, K)

def compute_torus2pers_map(
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    pano_height,
    pano_width,
    lonlat_to_xy=lonlat_to_xy,
    in_out_ratio = 3,
    **kwargs
):
    """
    Compute the XY mapping from the torus to perspective view.
    """
    fovy = fov
    fovx = fov * 0.5 * (1 - 1/in_out_ratio)

    x, y = torch.meshgrid(
        torch.linspace(-0.5, 0.5, pers_width),
        torch.linspace(-0.5, 0.5, pers_height),
        indexing="xy",
    )
    lat = phi + y * fovy
    stretch = 2 * in_out_ratio / (in_out_ratio + 1 + (in_out_ratio - 1) * torch.cos(torch.deg2rad(lat)))
    lon = theta + x * fovx * stretch
    lonlat = torch.stack([lon, lat], dim=-1)
    # wrapping
    lonlat[..., 0] = (lonlat[..., 0] + 180) % 360 - 180
    lonlat[..., 1] = (lonlat[..., 1] + 180) % 360 - 180

    return lonlat_to_xy(lonlat, pano_height, pano_width, lonrange=(-180, 180), latrange=(-180, 180))


def compute_pers2torus_map(
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    pano_height,
    pano_width,
    xy_to_lonlat=xy_to_lonlat,
    in_out_ratio = 3,
    **kwargs
):
    """
    Compute the inverse XY mapping from the perspective view to panorama.
    """
    fovy = fov
    fovx = fov * 0.5 * (1 - 1/in_out_ratio)

    x, y = torch.meshgrid(
        torch.arange(pano_width), torch.arange(pano_height), indexing="xy"
    )
    lonlat = xy_to_lonlat(torch.stack([x, y], dim=-1), pano_height, pano_width, lonrange=(-180, 180), latrange=(-180, 180))
    # wrapping
    lonlat[..., 0] = (lonlat[..., 0] + 180 - theta) % 360 - 180 + theta
    lonlat[..., 1] = (lonlat[..., 1] + 180 - phi) % 360 - 180 + phi

    y = pers_height * ((lonlat[..., 1] - phi) / fovy + 0.5)
    stretch = (in_out_ratio + 1 + (in_out_ratio - 1) * torch.cos(torch.deg2rad(lonlat[..., 1]))) / (2 * in_out_ratio)
    x = pers_width * (stretch * (lonlat[..., 0] - theta) / (fovx) + 0.5)

    return torch.stack([x, y], dim=-1)

def pano_to_pers(panorama, pano2pers, mode="nearest"):
    perspective_image = remap(panorama, pano2pers, mode)
    return perspective_image


def pano_to_pers_raw(
    panorama,
    fov,
    theta,
    phi,
    pers_height,
    pers_width,
    mode="nearest",
    mapping_func = compute_pano2pers_map,
    **kwargs
):
    """
    Transform a panorama image to a perspective view.
    """
    pano_height, pano_width = panorama.shape[-2], panorama.shape[-1]
    pano2pers = mapping_func(
        fov, theta, phi, pers_height, pers_width, pano_height, pano_width, **kwargs
    )
    pano2pers = pano2pers.to(panorama.device)
    return pano_to_pers(panorama, pano2pers, mode)


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
    perspective,
    fov,
    theta,
    phi,
    pano_height,
    pano_width,
    return_mask=False,
    mode="nearest",
    mapping_func = compute_pers2pano_map,
    **kwargs
):
    """
    Transform a perspective view image to a panorama.
    """
    pers_height, pers_width = perspective.shape[-2], perspective.shape[-1]
    pers2pano = mapping_func(
        fov, theta, phi, pers_height, pers_width, pano_height, pano_width, **kwargs
    )
    pers2pano = pers2pano.to(perspective.device)
    return pers_to_pano(perspective, pers2pano, return_mask, mode)


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