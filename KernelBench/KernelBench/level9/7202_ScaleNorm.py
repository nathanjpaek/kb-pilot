import math
import torch
import torch.nn as nn


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    """All g’s in SCALE NORM are initialized to sqrt(d)"""

    def __init__(self, scale, eps=1e-05):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=
            self.eps)
        return x * norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
