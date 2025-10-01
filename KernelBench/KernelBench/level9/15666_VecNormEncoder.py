import torch
import torch.nn as nn
import torch.nn.functional as F


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


class VecNormEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3):
        super().__init__(N_joints, N_dims)

    @property
    def encoder_name(self):
        return 'VecNorm'

    def forward(self, vecs, refs=None, *args, **kwargs):
        """
        Args:
          vecs (N_rays, *, ...): vector to normalize.
          refs (N_rays, N_pts, ...): reference tensor for shape expansion.
        Returns:
          (N_rays, N_pts, *): normalized vector with expanded shape (if needed).
        """
        n = F.normalize(vecs, dim=-1, p=2).flatten(start_dim=2)
        if refs is not None:
            n = n.expand(*refs.shape[:2], -1)
        return n


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
