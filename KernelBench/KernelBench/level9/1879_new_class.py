import torch
from torch import nn


class new_class(nn.Module):

    def __init__(self):
        super(new_class, self).__init__()

    def forward(self, input):
        return input + 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
