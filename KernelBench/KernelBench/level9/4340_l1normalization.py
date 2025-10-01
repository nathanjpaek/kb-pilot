import torch
import torch.nn as nn


class l1normalization(nn.Module):

    def __init__(self, scale):
        super(l1normalization, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        return self.scale * x * x.pow(1).sum(dim).clamp(min=1e-12).rsqrt(
            ).expand_as(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
