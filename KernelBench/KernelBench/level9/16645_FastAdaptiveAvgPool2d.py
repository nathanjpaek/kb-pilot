import torch
import torch.nn as nn
import torch.nn.parallel
import torch._utils
import torch.optim


class FastAdaptiveAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastAdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3)) if self.flatten else x.mean((2, 3), keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
