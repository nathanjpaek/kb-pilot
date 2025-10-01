import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class FreqUpsample(nn.Module):

    def __init__(self, factor: 'int', mode='nearest'):
        super().__init__()
        self.f = float(factor)
        self.mode = mode

    def forward(self, x: 'Tensor') ->Tensor:
        return F.interpolate(x, scale_factor=[1.0, self.f], mode=self.mode)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'factor': 4}]
