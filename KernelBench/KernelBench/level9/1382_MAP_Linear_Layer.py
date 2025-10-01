import torch
import numpy as np
import torch.nn as nn


class MAP_Linear_Layer(nn.Module):

    def __init__(self, n_input, n_output):
        super(MAP_Linear_Layer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_input, n_output).normal_(
            0, 1 / np.sqrt(4 * n_output)))
        self.bias = nn.Parameter(torch.Tensor(n_output).normal_(0, 1e-10))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input': 4, 'n_output': 4}]
