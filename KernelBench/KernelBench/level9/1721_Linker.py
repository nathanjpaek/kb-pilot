import torch
import torch.nn as nn
import torch.nn.functional as F


class Linker(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size, strides):
        super(Linker, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, strides)
        self.pad = (0, 0, 0, 0) + ((outplanes - inplanes) // 2,) * 2

    def forward(self, x):
        x = self.avgpool(x)
        x = F.pad(x, self.pad, 'constant', 0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'outplanes': 4, 'kernel_size': 4, 'strides': 1}
        ]
