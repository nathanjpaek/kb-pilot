import torch
from torch import nn
import torch.utils.data.distributed


class LipschitzCube(nn.Module):

    def forward(self, x):
        return (x >= 1) * (x - 2 / 3) + (x <= -1) * (x + 2 / 3) + (x > -1) * (x
             < 1) * x ** 3 / 3


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
