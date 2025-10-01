import torch
import numpy as np
import torch.nn as nn


def calculate_angle(a, b=None):
    if b is None:
        b = torch.Tensor([0.0, 0.0, 1.0]).view(1, 1, -1)
    dot_product = (a * b).sum(-1)
    norm_a = torch.norm(a, p=2, dim=-1)
    norm_b = torch.norm(b, p=2, dim=-1)
    cos = dot_product / (norm_a * norm_b)
    cos = torch.clamp(cos, -1.0 + 1e-06, 1.0 - 1e-06)
    angle = torch.acos(cos)
    assert not torch.isnan(angle).any()
    return angle - 0.5 * np.pi


class BaseEncoder(nn.Module):

    def __init__(self, N_joints=24, N_dims=None):
        super().__init__()
        self.N_joints = N_joints
        self.N_dims = N_dims if N_dims is not None else 1

    @property
    def dims(self):
        return self.N_joints * self.N_dims

    @property
    def encoder_name(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class RayAngEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=1):
        super().__init__(N_joints, N_dims)

    @property
    def encoder_name(self):
        return 'RayAng'

    def forward(self, rays_t, pts_t, *args, **kwargs):
        """
        Args:
          rays_t (N_rays, 1, N_joints, 3): rays direction in local space (joints at (0, 0, 0))
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0))
        Returns:
          d (N_rays, N_pts, N_joints*3): normalized ray direction in local space
        """
        return calculate_angle(pts_t, rays_t)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
