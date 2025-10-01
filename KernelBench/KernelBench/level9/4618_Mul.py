import torch
import torch.nn as nn


class Mul(nn.Module):

    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x):
        x = torch.mul(x, 20)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
