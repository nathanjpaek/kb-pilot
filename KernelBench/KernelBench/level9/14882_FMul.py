import torch
import torch.nn as nn


class FMul(nn.Module):

    def __init__(self):
        super(FMul, self).__init__()

    def forward(self, x, y):
        x = x * y
        x = x * 10.0
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
