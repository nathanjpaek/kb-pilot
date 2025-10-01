import torch
from itertools import product as product
import torch.nn as nn


class fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand_planes, st=1):
        super(fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1,
            stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, int(expand_planes / 2),
            kernel_size=1, stride=st)
        self.conv3 = nn.Conv2d(squeeze_planes, int(expand_planes / 2),
            kernel_size=3, stride=st, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out2 = self.conv3(x)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'squeeze_planes': 4, 'expand_planes': 4}]
