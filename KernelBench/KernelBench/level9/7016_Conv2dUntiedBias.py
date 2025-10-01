import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class Conv2dUntiedBias(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, input_len,
        stride=1, padding=0, dilation=1, groups=1):
        super(Conv2dUntiedBias, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels //
            groups, *kernel_size))
        height = 1
        width = self.calc_output_width(input_len, kernel_size)
        self.bias = nn.Parameter(torch.Tensor(out_channels, height, width))
        self.reset_parameters()

    def calc_output_width(self, input_length, kernel_size, stride=1):
        return (input_length - kernel_size[-1] + stride) // stride

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = F.conv2d(input, self.weight, None, self.stride, self.
            padding, self.dilation, self.groups)
        output += self.bias.unsqueeze(0).repeat(input.size(0), 1, 1, 1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'input_len': 4}]
