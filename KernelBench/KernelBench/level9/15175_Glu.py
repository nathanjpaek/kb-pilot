import torch
import torch.nn as nn


class Glu(nn.Module):

    def __init__(self, dim):
        super(Glu, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_in, x_gate = x.chunk(2, dim=self.dim)
        return x_in * x_gate.sigmoid()


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
