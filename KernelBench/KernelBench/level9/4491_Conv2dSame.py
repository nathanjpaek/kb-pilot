import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
import torch.nn.functional as F
from typing import Optional
from typing import Tuple
import torch.nn.parallel
import torch.optim


def _calc_same_pad(input_: 'int', kernel: 'int', stride: 'int', dilation: 'int'
    ):
    """calculate same padding"""
    return max((-(input_ // -stride) - 1) * stride + (kernel - 1) *
        dilation + 1 - input_, 0)


def conv2d_same(input_, weight: 'torch.Tensor', bias:
    'Optional[torch.Tensor]'=None, stride: 'Tuple[int, int]'=(1, 1),
    padding: 'Tuple[int, int]'=(0, 0), dilation: 'Tuple[int, int]'=(1, 1),
    groups: 'int'=1):
    """conv2d with same padding"""
    input_height, input_width = input_.size()[-2:]
    kernel_height, kernel_width = weight.size()[-2:]
    pad_h = _calc_same_pad(input_height, kernel_height, stride[0], dilation[0])
    pad_w = _calc_same_pad(input_width, kernel_width, stride[1], dilation[1])
    input_ = F.pad(input_, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, 
        pad_h - pad_h // 2])
    return F.conv2d(input_, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias)

    def forward(self, input_):
        return conv2d_same(input_, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
