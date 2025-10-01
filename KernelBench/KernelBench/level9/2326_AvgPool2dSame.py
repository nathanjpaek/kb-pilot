import math
import torch
import numpy as np
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


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


def to_2tuple(item):
    if np.isscalar(item):
        return item, item
    else:
        return item


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """

    def __init__(self, kernel_size: 'int', stride=None, padding=0,
        ceil_mode=False, count_include_pad=True):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0),
            ceil_mode, count_include_pad)

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding,
            self.ceil_mode, self.count_include_pad)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
