from torch.nn import Module
import torch
from torch.nn import Conv2d


class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        return self.conv_r(input_r) - self.conv_i(input_i), self.conv_r(input_i
            ) + self.conv_i(input_r)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
