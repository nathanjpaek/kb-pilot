import torch
from torch import nn
import torch.nn.functional as F


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (l_out - 1) * stride - l_in + dilation * (kernel - 1) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])
    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
        padding=padding // 2, dilation=dilation, groups=groups)


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.
            stride, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
