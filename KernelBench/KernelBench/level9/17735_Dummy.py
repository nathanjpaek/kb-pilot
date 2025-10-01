import torch
from torch import nn


class Dummy(nn.Module):

    def forward(self, input):
        x = input
        return x + 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
