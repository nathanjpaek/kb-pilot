import torch
import torch.nn as nn


class FDiv(nn.Module):

    def __init__(self):
        super(FDiv, self).__init__()

    def forward(self, x, y):
        x = x / 2
        y = y / 2
        x = x / y
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
