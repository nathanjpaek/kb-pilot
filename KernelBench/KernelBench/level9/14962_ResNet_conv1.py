import math
import torch
import torch.utils.data
import torch.nn as nn


class ResNet_conv1(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_conv1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
            bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = x.clone()
        y[:, 0, :, :] = (y[:, 0, :, :] - 0.485) / 0.229
        y[:, 1, :, :] = (y[:, 1, :, :] - 0.485) / 0.224
        y[:, 2, :, :] = (y[:, 2, :, :] - 0.485) / 0.224
        x = self.conv1(y)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'block': 4, 'layers': 1}]
