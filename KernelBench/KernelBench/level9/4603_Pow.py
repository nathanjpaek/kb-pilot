import torch
import torch.nn as nn


class Pow(nn.Module):

    def __init__(self):
        super(Pow, self).__init__()

    def forward(self, x):
        x = torch.pow(x, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
