import torch
import torch.nn as nn
import torch.nn


class C3D(nn.Module):

    def __init__(self, inplanes, planes):
        super(C3D, self).__init__()
        self.c3d = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.c3d(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
