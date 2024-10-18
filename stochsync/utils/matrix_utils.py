import torch
#from jaxtyping import Float

# dummy type for type hinting
class Float:
    def __class_getitem__(cls, item):
        return torch.Tensor

def quat_to_rot(quat: Float[torch.Tensor, "4"]) -> Float[torch.Tensor, "3 3"]:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat: A tensor representing a quaternion (w, x, y, z).

    Returns:
        A tensor representing the corresponding rotation matrix.
    """
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return torch.Tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2],
    ])

def quats_to_rots(quat: Float[torch.Tensor, "B 4"]) -> Float[torch.Tensor, "B 3 3"]:
    """
    Converts a batch of quaternions to a batch of rotation matrices.

    Args:
        quat: A tensor representing a batch of B quaternions (w, x, y, z).

    Returns:
        A tensor representing the corresponding batch of B rotation matrices.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([
        1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
        2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w,
        2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2,
    ], dim=1).view(-1, 3, 3)

def rot_to_quat(rotation_matrix: Float[torch.Tensor, "3 3"], eps: float = 1e-6) -> Float[torch.Tensor, "4"]:
    """
    Converts a rotation matrix to a quaternion.

    Args:
        rotation_matrix: A tensor representing a rotation matrix.
        eps: A small value to prevent division by zero. Default is 1e-6.

    Returns:
        A tensor representing the corresponding quaternion (w, x, y, z).
    """
    trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
    quaternion = torch.zeros((4,)).to(rotation_matrix.device)
    quaternion[0] = torch.sqrt(1 + trace + eps) / 2
    divider = 4 * quaternion[0]
    divider[divider.abs() < eps] = eps

    quaternion[1] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / divider
    quaternion[2] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / divider
    quaternion[3] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / divider

    return quaternion

def rots_to_quats(rotation_matrix: Float[torch.Tensor, "B 3 3"], eps: float = 1e-6) -> Float[torch.Tensor, "B 4"]:
    """
    Converts a B of rotation matrices to quaternions.

    Args:
        rotation_matrix: A tensor representing a batch of B rotation matrices.
        eps: A small value to prevent division by zero. Default is 1e-6.

    Returns:
        A tensor representing the corresponding quaternions (w, x, y, z).
    """
    B = rotation_matrix.shape[0]
    trace = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]
    quaternion = torch.zeros((B, 4)).to(rotation_matrix.device)
    quaternion[:, 0] = torch.sqrt(1 + trace + eps) / 2
    divider = 4 * quaternion[:, 0]
    divider[divider.abs() < eps] = eps

    quaternion[:, 1] = (rotation_matrix[:, 2, 1] - rotation_matrix[:, 1, 2]) / divider
    quaternion[:, 2] = (rotation_matrix[:, 0, 2] - rotation_matrix[:, 2, 0]) / divider
    quaternion[:, 3] = (rotation_matrix[:, 1, 0] - rotation_matrix[:, 0, 1]) / divider

    return quaternion

def make_quat(axis: Float[torch.Tensor, "3"], angle: float) -> Float[torch.Tensor, "4"]:
    """
    Creates a quaternion from an axis and an angle.

    Args:
        axis: A tensor representing the axis of rotation.
        angle: The rotation angle in radians.

    Returns:
        A tensor representing the quaternion (w, x, y, z).
    """
    if isinstance(angle, float):
        angle = torch.tensor([angle])

    axis = torch.nn.functional.normalize(axis, dim=0)
    w = torch.cos(angle / 2)
    x, y, z = torch.sin(angle / 2) * axis
    q = torch.tensor([w, x, y, z])
    return q

def multiply_quat(q1: Float[torch.Tensor, "4"], q2: Float[torch.Tensor, "4"]) -> Float[torch.Tensor, "4"]:
    """
    Multiplies two quaternions.

    Args:
        q1: A tensor representing the first quaternion (w, x, y, z).
        q2: A tensor representing the second quaternion (w, x, y, z).

    Returns:
        A tensor representing the product of the two quaternions (w, x, y, z).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q3 = torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
    return q3

def multiply_quats(q1: Float[torch.Tensor, "B 4"], q2: Float[torch.Tensor, "B 4"]) -> Float[torch.Tensor, "B 4"]:
    """
    Multiplies a batch of quaternions.

    Args:
        q1: A tensor representing the first batch of B quaternions (w, x, y, z).
        q2: A tensor representing the second batch of B quaternions (w, x, y, z).

    Returns:
        A tensor representing the product of the batches of quaternions (w, x, y, z).
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    q3 = torch.zeros_like(q1)
    q3[:, 0] = w1*w2 - x1*x2 - y1*y2 - z1*z2
    q3[:, 1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
    q3[:, 2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
    q3[:, 3] = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return q3

def rodrigues(axis: Float[torch.Tensor, "3"], angle: float) -> Float[torch.Tensor, "3 3"]:
    """
    Generates a rotation matrix from an axis and an angle.

    Args:
        axis: A tensor representing the axis of rotation.
        angle: The rotation angle in radians.

    Returns:
        A tensor representing the rotation matrix.
    """
    if isinstance(angle, float) or isinstance(angle, int):
        angle = torch.tensor([angle], dtype=torch.float)

    axis = torch.nn.functional.normalize(axis, dim=0)
    x, y, z = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1 - c

    return torch.tensor([
        [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])

def apply_projection(matrix, points):
    is_1D = points.dim() == 1

    if is_1D:
        points = points.unsqueeze(0)

    points_t = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1).t()
    points_t = matrix @ points_t
    points_t /= points_t[3].clone()
    
    if is_1D:
        return points_t[:3].t().squeeze(0)
    return points_t[:3].t()