import torch
from torch import nn


class StdConv(nn.Conv2d):

    def forward(self, x):
        return self._conv_forward(x, self.standarize(self.weight), self.bias)

    def standarize(self, x):
        return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2,
            3), keepdim=True) + 1e-06)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
