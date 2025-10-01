import math
import torch
import torch.nn as nn
import torch.utils.data._utils
import torch.nn
import torch.optim


class Embedder(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Embedder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, x):
        output = self.linear(x) * math.sqrt(self.dim_out)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
