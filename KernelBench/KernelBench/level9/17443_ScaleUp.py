import torch
import torch.nn as nn
from torch.nn import Parameter


class ScaleUp(nn.Module):
    """ScaleUp"""

    def __init__(self, scale):
        super(ScaleUp, self).__init__()
        self.scale = Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
