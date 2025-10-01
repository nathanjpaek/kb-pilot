import torch
from torch import nn


class ComplexMul(nn.Module):

    def forward(self, a, b):
        re = a[:, :1] * b[:, :1] - a[:, 1:] * b[:, 1:]
        im = a[:, :1] * b[:, 1:] + a[:, :1] * b[:, 1:]
        return torch.cat((re, im), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
