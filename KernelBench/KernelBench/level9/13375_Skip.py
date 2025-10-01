import torch
from torch import nn


class Skip(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Skip, self).__init__()
        assert C_out % C_in == 0, 'C_out must be divisible by C_in'
        self.repeats = 1, C_out // C_in, 1, 1

    def forward(self, x):
        return x.repeat(self.repeats)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C_in': 4, 'C_out': 4, 'stride': 1}]
