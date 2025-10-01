import torch
import torch.nn as nn


class ExpModule(nn.Module):

    def __init__(self):
        super(ExpModule, self).__init__()

    def forward(self, x):
        return torch.exp(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
