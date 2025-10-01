import torch
import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

    def forward(self, x):
        return x * 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
