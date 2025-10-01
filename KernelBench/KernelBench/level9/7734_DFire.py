import torch
from torch import nn


class DFire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes,
        expand3x3_planes):
        super(DFire, self).__init__()
        self.inplanes = inplanes
        self.expand1x1 = nn.Conv2d(inplanes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ELU(inplace=True)
        self.expand3x3 = nn.Conv2d(inplanes, expand3x3_planes, kernel_size=
            3, padding=1)
        self.expand3x3_activation = nn.ELU(inplace=True)
        self.squeeze = nn.Conv2d(expand3x3_planes + expand1x1_planes,
            squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ELU(inplace=True)

    def forward(self, x):
        x = torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.
            expand3x3_activation(self.expand3x3(x))], 1)
        x = self.squeeze_activation(self.squeeze(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4,
        'expand3x3_planes': 4}]
