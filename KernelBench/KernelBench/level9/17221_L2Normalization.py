import torch
import torch.nn as nn


class L2Normalization(nn.Module):

    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, x):
        div = torch.sqrt(torch.sum(x * x, 1))
        x = (x.T / (div + 1e-10)).T
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
