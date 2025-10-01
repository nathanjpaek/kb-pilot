import torch
from torch import nn


class ClampNorm(nn.Module):

    def __init__(self):
        super(ClampNorm, self).__init__()

    def forward(self, x):
        out = x.clamp(0.0, 1.0)
        return out / out.sum(1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
