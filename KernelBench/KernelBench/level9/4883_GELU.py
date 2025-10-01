import torch
from torch import nn


class GELU(nn.Module):

    def forward(self, x):
        return nn.functional.gelu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
