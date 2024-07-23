import torch

def create_random_pcd(radius, num_splats):
    """
    Initialize the splats with random points.
    """

    xyz = torch.rand(num_splats, 3) * 2 - 1  # [-1, 1]
    xyz = xyz * radius

    return xyz


def create_alpha_weight(height, width, safe_rad=0.1, device="cpu"):
    # Create a grid of coordinates
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Calculate the distance from the center
    distance = torch.sqrt(xx**2 + yy**2) - 0.1
    distance = torch.maximum(distance, torch.zeros_like(distance))
    distance = distance

    return distance