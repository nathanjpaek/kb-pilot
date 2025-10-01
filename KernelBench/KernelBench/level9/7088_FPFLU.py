import torch
from torch import nn


class FPFLU(nn.Module):

    def forward(self, x):
        return torch.maximum(x, x / (1 + x * x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
