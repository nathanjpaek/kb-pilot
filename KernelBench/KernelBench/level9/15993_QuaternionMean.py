import torch
from torch import nn as nn


def quaternions_to_group_matrix(q):
    """Normalises q and maps to group matrix."""
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([r * r - i * i - j * j + k * k, 2 * (r * i + j * k),
        2 * (r * j - i * k), 2 * (r * i - j * k), -r * r + i * i - j * j + 
        k * k, 2 * (i * j + r * k), 2 * (r * j + i * k), 2 * (i * j - r * k
        ), -r * r - i * i + j * j + k * k], -1).view(*q.shape[:-1], 3, 3)


class QuaternionMean(nn.Module):

    def __init__(self, input_dims):
        super().__init__()
        self.map = nn.Linear(input_dims, 4)

    def forward(self, x):
        return quaternions_to_group_matrix(self.map(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dims': 4}]
