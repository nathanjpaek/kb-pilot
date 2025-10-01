import torch
from torch import nn
from collections import OrderedDict


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes,
        expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.group1 = nn.Sequential(OrderedDict([('squeeze', nn.Conv2d(
            inplanes, squeeze_planes, kernel_size=1)), (
            'squeeze_activation', nn.ReLU(inplace=True))]))
        self.group2 = nn.Sequential(OrderedDict([('expand1x1', nn.Conv2d(
            squeeze_planes, expand1x1_planes, kernel_size=1)), (
            'expand1x1_activation', nn.ReLU(inplace=True))]))
        self.group3 = nn.Sequential(OrderedDict([('expand3x3', nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)), (
            'expand3x3_activation', nn.ReLU(inplace=True))]))

    def forward(self, x):
        x = self.group1(x)
        return torch.cat([self.group2(x), self.group3(x)], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4,
        'expand3x3_planes': 4}]
