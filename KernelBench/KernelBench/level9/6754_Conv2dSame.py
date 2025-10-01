import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
from typing import Optional
from typing import Tuple
from torch.jit.annotations import List


def get_same_padding(x: 'int', k: 'int', s: 'int', d: 'int'):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k: 'List[int]', s: 'List[int]', d: 'List[int]'=(1, 1),
    value: 'float'=0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw,
        k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - 
            pad_h // 2], value=value)
    return x


def conv2d_same(x, weight: 'torch.Tensor', bias: 'Optional[torch.Tensor]'=
    None, stride: 'Tuple[int, int]'=(1, 1), padding: 'Tuple[int, int]'=(0, 
    0), dilation: 'Tuple[int, int]'=(1, 1), groups: 'int'=1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels,
            kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
