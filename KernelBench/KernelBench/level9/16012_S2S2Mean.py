import torch
from torch import nn as nn


def s2s2_gram_schmidt(v1, v2):
    """Normalise 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix."""
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-05)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-05)
    e3 = torch.cross(e1, e2)
    return torch.stack([e1, e2, e3], 1)


class S2S2Mean(nn.Module):
    """Module to map R^6 -> SO(3) with S2S2 method."""

    def __init__(self, input_dims):
        super().__init__()
        self.map = nn.Linear(input_dims, 6)
        self.map.weight.data.uniform_(-10, 10)
        self.map.bias.data.uniform_(-10, 10)

    def forward(self, x):
        v = self.map(x).double().view(-1, 2, 3)
        v1, v2 = v[:, 0], v[:, 1]
        return s2s2_gram_schmidt(v1, v2).float()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dims': 4}]
