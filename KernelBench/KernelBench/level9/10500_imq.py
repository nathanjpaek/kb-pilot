import torch
import torch as tr
import torch.nn as nn


class imq(nn.Module):

    def __init__(self, c=1.0):
        super(imq, self).__init__()
        self.c = c

    def forward(self, x: 'tr.Tensor'):
        return 1 / (self.c ** 2 + x ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
