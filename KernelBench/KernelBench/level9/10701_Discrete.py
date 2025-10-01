import torch
import torch.nn as nn


class Discrete(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.softmax(x, dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
