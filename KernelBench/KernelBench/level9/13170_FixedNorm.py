import torch
import torch.nn as nn


class FixedNorm(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1.0 / 2)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return x_normed


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4}]
