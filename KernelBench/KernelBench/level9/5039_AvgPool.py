import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data


class AvgPool(nn.Module):
    """1-d average pooling module."""

    def __init__(self, stride=None, padding=0):
        super(AvgPool, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        kernel_size = x.size(2)
        x = F.max_pool1d(input=x, kernel_size=kernel_size, stride=self.
            stride, padding=self.padding)
        return x.squeeze(dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
