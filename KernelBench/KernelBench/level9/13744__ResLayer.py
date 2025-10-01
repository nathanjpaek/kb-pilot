import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResLayer(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, act='tanh'):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=True)
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=True)
        if act == 'tanh':
            self.act = F.tanh
        elif act == 'relu':
            self.act = F.relu

    def forward(self, x):
        res = x
        out = self.fc1(self.act(x))
        out = self.fc2(self.act(out))
        return res + out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4, 'dim_hidden': 4}]
