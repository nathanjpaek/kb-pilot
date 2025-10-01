import torch
import torch as th
from torch import nn
import torch.random
import torch


class ZeroModule(nn.Module):
    """Module that always returns zeros of same shape as input."""

    def __init__(self, features_dim: 'int'):
        """Builds ZeroModule."""
        super().__init__()
        self.features_dim = features_dim

    def forward(self, x: 'th.Tensor') ->th.Tensor:
        """Returns zeros of same shape as `x`."""
        assert x.shape[1:] == (self.features_dim,)
        return th.zeros_like(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'features_dim': 4}]
