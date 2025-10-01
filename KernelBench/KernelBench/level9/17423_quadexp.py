import torch
import torch as tr
import torch.nn as nn


class quadexp(nn.Module):

    def __init__(self, sigma=2.0):
        super(quadexp, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        return tr.exp(-x ** 2 / self.sigma ** 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
