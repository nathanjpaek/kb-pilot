import torch
from torch import nn


class _ConvReLU_(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, relu=True):
        super(_ConvReLU_, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=padding, dilation=dilation, bias=False))
        """
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
            ),
        )
        """
        if relu:
            self.add_module('relu', nn.ReLU())

    def forward(self, x):
        return super(_ConvReLU_, self).forward(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4, 'dilation': 1}]
