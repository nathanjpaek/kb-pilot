from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair


class MaskedConv(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
        super(MaskedConv, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.use_bias = bias
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available(
            ) else torch.FloatTensor
        self.weight = nn.Parameter(self.floatTensor(out_channels,
            in_channels, *self.kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(self.floatTensor(out_channels),
                requires_grad=True)
        else:
            self.bias = None
        self.mask = nn.Parameter(self.floatTensor(out_channels, in_channels,
            *self.kernel_size), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.ones_(self.mask)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride,
            self.padding, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
