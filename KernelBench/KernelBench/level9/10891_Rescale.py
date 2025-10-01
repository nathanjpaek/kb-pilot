import torch
import torch.nn as nn


class Rescale(nn.Module):

    def __init__(self, sign):
        super(Rescale, self).__init__()
        rgb_mean = 0.4488, 0.4371, 0.404
        bias = sign * torch.Tensor(rgb_mean).reshape(1, 3, 1, 1)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return x + self.bias


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {'sign': 4}]
