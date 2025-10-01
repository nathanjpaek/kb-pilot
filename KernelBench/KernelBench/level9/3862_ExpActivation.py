import torch
import torch.nn as nn


class ExpActivation(nn.Module):

    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(-x ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
