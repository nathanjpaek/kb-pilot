import torch
import torch.nn as nn
import torch.nn.functional as F


class first_conv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'
        self.transform = None

    def forward(self, x):
        restore_w = self.weight
        max = restore_w.data.max()
        weight_q = restore_w.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - restore_w).detach() + restore_w
        return F.conv2d(x, weight_q, self.bias, self.stride, self.padding,
            self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
