import torch
import torch.nn as nn


class Bias(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
