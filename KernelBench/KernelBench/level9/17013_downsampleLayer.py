import torch
import torch.nn as nn


class downsampleLayer(nn.Module):
    """
    A downsample layer of UNet. LeakyReLU is used as the activation func. 
    """

    def __init__(self, infeature, outfeature, kernelSize, strides=2,
        paddings=1, bn=False):
        super(downsampleLayer, self).__init__()
        self.conv = nn.Conv2d(infeature, outfeature, kernelSize, stride=
            strides, padding=paddings)
        self.acti = nn.LeakyReLU(negative_slope=0.2)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(outfeature, momentum=0.8)

    def forward(self, x):
        y = self.acti(self.conv(x))
        if self.bn is not None:
            y = self.bn(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'infeature': 4, 'outfeature': 4, 'kernelSize': 4}]
