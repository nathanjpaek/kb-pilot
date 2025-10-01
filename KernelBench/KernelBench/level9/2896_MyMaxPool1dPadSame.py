import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from itertools import product as product
from math import sqrt as sqrt


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size, stride_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), 'constant', 0)
        net = self.max_pool(net)
        return net


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'stride_size': 1}]
