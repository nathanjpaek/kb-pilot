import torch
from torch import nn


class BiAvg(nn.AvgPool1d):

    def forward(self, x):
        x = x.transpose(1, 2)
        x = super().forward(x)
        return x.transpose(1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
