import torch
import torch.nn as nn


class Scale2D(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.register_parameter('alpha', torch.nn.Parameter(torch.ones([1,
            n, 1, 1])))
        self.register_parameter('beta', torch.nn.Parameter(torch.ones([1, n,
            1, 1])))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4}]
