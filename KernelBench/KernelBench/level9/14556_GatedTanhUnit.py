import torch
import torch.nn as nn


def gated_tanh(x, dim):
    """Gated Tanh activation."""
    x_tanh, x_sigmoid = torch.chunk(x, 2, dim=dim)
    return torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)


class GatedTanhUnit(nn.Module):
    """Gated Tanh activation."""

    def __init__(self, dim=-1):
        super(GatedTanhUnit, self).__init__()
        self.dim = dim

    def forward(self, x):
        return gated_tanh(x, dim=self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
