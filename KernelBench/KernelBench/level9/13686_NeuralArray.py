import torch
import torch.utils.data
import torch
import torch.nn as nn


class NeuralArray(nn.Module):

    def __init__(self, dim, random_init=False):
        super(NeuralArray, self).__init__()
        self.dim = dim
        if random_init:
            self.register_parameter('data', torch.nn.Parameter(torch.randn(
                self.dim, requires_grad=True)))
        else:
            self.register_parameter('data', torch.nn.Parameter(torch.zeros(
                self.dim, requires_grad=True)))

    def forward(self, id):
        return self.data[id]

    def regularizer_zero(self):
        return torch.mean(torch.pow(self.data, 2.0))


def get_inputs():
    return [torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'dim': 4}]
