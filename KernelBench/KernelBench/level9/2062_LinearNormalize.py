import torch
from torch import nn


class LinearNormalize(nn.Module):

    def forward(self, x):
        return (x - x.min()) / x.max()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
