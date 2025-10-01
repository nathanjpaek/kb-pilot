import torch
import torch.nn as nn
import torch.nn.functional as F


class USConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, us=[False, False]):
        super(USConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.width_mult = None
        self.us = us

    def forward(self, inputs):
        in_channels = inputs.shape[1] // self.groups if self.us[0
            ] else self.in_channels // self.groups
        out_channels = int(self.out_channels * self.width_mult) if self.us[1
            ] else self.out_channels
        weight = self.weight[:out_channels, :in_channels, :, :]
        bias = self.bias[:out_channels] if self.bias is not None else self.bias
        y = F.conv2d(inputs, weight, bias, self.stride, self.padding, self.
            dilation, self.groups)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
