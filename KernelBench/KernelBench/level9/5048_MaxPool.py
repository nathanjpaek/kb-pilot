import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data


class MaxPool(nn.Module):
    """1-d max-pooling module."""

    def __init__(self, stride=None, padding=0, dilation=1):
        super(MaxPool, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        kernel_size = x.size(2)
        x = F.max_pool1d(input=x, kernel_size=kernel_size, stride=self.
            stride, padding=self.padding, dilation=self.dilation)
        return x.squeeze(dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
