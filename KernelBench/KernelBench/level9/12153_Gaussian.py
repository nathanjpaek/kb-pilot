import torch
import torch.nn as nn


class Gaussian(nn.Module):

    def forward(self, x):
        return torch.exp(-x * x / 2.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
