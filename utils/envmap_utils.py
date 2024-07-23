from math import tan, pi

from PIL import Image
import torch
from k_utils.image_utils import torch_to_pil, pil_to_torch
from Utils.matrix_utils import rodrigues

def xyz2lonlat(xyz):
    atan2 = torch.arctan2
    asin = torch.arcsin

    norm = torch.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = torch.stack(lst, dim=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., :1] / (2 * pi) + 0.5) * 2 - 1
    Y = (lonlat[..., 1:] / (pi) + 0.5) * 2 - 1
    lst = [X, Y]

    out = torch.stack(lst, dim=-1).squeeze()

    return out 

class Equirectangular:
    def __init__(self, img_name):
        # as RGB
        self._img = Image.open(img_name).convert('RGB')
        self._img_tensor = pil_to_torch(self._img).to('cuda')
        self._height, self._width = self._img.size

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        f = 0.5 * width * 1 / tan(0.5 * FOV * pi / 180.0)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = torch.tensor([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], dtype = torch.float32, device='cuda')
        K_inv = torch.inverse(K)
        
        x = torch.arange(width, dtype=torch.float32, device='cuda')
        y = torch.arange(height, dtype=torch.float32, device='cuda')
        y, x = torch.meshgrid(y, x)
        z = torch.ones_like(x, dtype=torch.float32, device='cuda')
        xyz = torch.stack([x, y, z], dim=-1)
        xyz = xyz @ K_inv.T

        y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        R1 = rodrigues(y_axis, THETA * pi / 180.0)
        R2 = rodrigues(R1 @ x_axis, PHI * pi / 180.0)
        R = (R2 @ R1).to('cuda')
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=(self._height, self._width)).to(torch.float32)
        
        persp = torch.nn.functional.grid_sample(self._img_tensor, XY.unsqueeze(0), mode='bilinear')

        return persp

# eq = Equirectangular('envmap.png')
# persp = eq.GetPerspective(72, 30, 0, 768, 1024)
# torch_to_pil(persp)