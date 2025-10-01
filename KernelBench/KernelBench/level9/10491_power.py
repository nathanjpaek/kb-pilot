import torch
import torch as tr
import torch.nn as nn


class power(nn.Module):

    def __init__(self):
        super(power, self).__init__()

    def forward(self, x: 'tr.Tensor'):
        return x.pow(2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
