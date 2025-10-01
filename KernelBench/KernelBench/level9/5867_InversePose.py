import torch
import torch.nn as nn


def inverse_pose(pose, eps=1e-06):
    """Function that inverts a 4x4 pose.

    Args:
        points (Tensor): tensor with poses.

    Returns:
        Tensor: tensor with inverted poses.

    Shape:
        - Input: :math:`(N, 4, 4)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pose = torch.rand(1, 4, 4)         # Nx4x4
        >>> pose_inv = tgm.inverse_pose(pose)  # Nx4x4
    """
    if not torch.is_tensor(pose):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(pose)))
    if not len(pose.shape) == 3 and pose.shape[-2:] == (4, 4):
        raise ValueError('Input size must be a Nx4x4 tensor. Got {}'.format
            (pose.shape))
    r_mat = pose[..., :3, 0:3]
    t_vec = pose[..., :3, 3:4]
    r_mat_trans = torch.transpose(r_mat, 1, 2)
    pose_inv = pose.new_zeros(pose.shape) + eps
    pose_inv[..., :3, 0:3] = r_mat_trans
    pose_inv[..., :3, 3:4] = torch.matmul(-1.0 * r_mat_trans, t_vec)
    pose_inv[..., 3, 3] = 1.0
    return pose_inv


class InversePose(nn.Module):
    """Creates a transformation that inverts a 4x4 pose.

    Args:
        points (Tensor): tensor with poses.

    Returns:
        Tensor: tensor with inverted poses.

    Shape:
        - Input: :math:`(N, 4, 4)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> pose = torch.rand(1, 4, 4)  # Nx4x4
        >>> transform = tgm.InversePose()
        >>> pose_inv = transform(pose)  # Nx4x4
    """

    def __init__(self):
        super(InversePose, self).__init__()

    def forward(self, input):
        return inverse_pose(input)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
