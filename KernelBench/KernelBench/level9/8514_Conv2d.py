import torch
import numpy as np
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def get_causal_padding(kernel_size, strides, dilation_rate, n_dims=2):
    p_ = []
    for i in range(n_dims - 1, -1, -1):
        if strides[i] > 1 and dilation_rate[i] > 1:
            raise ValueError("can't have the stride and dilation over 1")
        p = (kernel_size[i] - strides[i]) * dilation_rate[i]
        p_ += p, 0
    return p_


def get_same_padding(kernel_size, strides, dilation_rate, n_dims=2):
    p_ = []
    for i in range(n_dims - 1, -1, -1):
        if strides[i] > 1 and dilation_rate[i] > 1:
            raise ValueError("Can't have the stride and dilation rate over 1")
        p = (kernel_size[i] - strides[i]) * dilation_rate[i]
        if p % 2 == 0:
            p = p // 2, p // 2
        else:
            p = int(np.ceil(p / 2)), int(np.floor(p / 2))
        p_ += p
    return tuple(p_)


def get_valid_padding(n_dims=2):
    p_ = (0,) * 2 * n_dims
    return p_


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding='same', dilation=1, *args, **kwargs):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(dilation, int):
            dilation = (dilation,) * 2
        self.stride = stride
        self.padding_str = padding.upper()
        if self.padding_str == 'SAME':
            self.pad_values = get_same_padding(kernel_size, stride, dilation)
        elif self.padding_str == 'VALID':
            self.pad_values = get_valid_padding()
        elif self.padding_str == 'CAUSAL':
            self.pad_values = get_causal_padding(kernel_size, stride, dilation)
        else:
            raise ValueError
        self.condition = np.sum(self.pad_values) != 0
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
            stride, 0, dilation, *args, **kwargs)

    def reset_parameters(self) ->None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x):
        if self.condition:
            x = F.pad(x, self.pad_values)
        x = super(Conv2d, self).forward(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
