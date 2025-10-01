import torch
import torch.nn as nn


class MyLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.01):
        super(MyLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.clamp(x, min=0.0) + torch.clamp(x, max=0.0
            ) * self.negative_slope


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
