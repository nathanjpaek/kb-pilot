import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim


class SeparableConvBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(inplanes, inplanes, kernel_size=3,
            stride=1, padding=1, groups=inplanes, bias=False)
        self.pointwise_conv = nn.Conv2d(inplanes, planes, kernel_size=1,
            stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
