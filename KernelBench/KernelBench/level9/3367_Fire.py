import math
import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1,
            stride=1)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1,
            stride=1)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3,
            stride=1, padding=1)
        self.relu2 = nn.ELU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

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
