import torch
import torch.nn as nn


class HadamardProduct(nn.Module):

    def __init__(self, shape):
        super(HadamardProduct, self).__init__()
        self.weights = nn.Parameter(torch.rand(shape))

    def forward(self, x):
        return x * self.weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'shape': 4}]
