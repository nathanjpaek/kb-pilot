from torch.nn import Module
import torch
import torch.nn as nn


class LinearActivation(Module):

    def __init__(self, in_features, out_features, act='gelu', bias=True):
        super(LinearActivation, self).__init__()
        self.Linear = nn.Linear(in_features, out_features, bias=bias)
        if act == 'relu':
            self.act_fn = nn.ReLU()
        elif act == 'tanh':
            self.act_fn = nn.Tanh()
        elif act == 'gelu':
            self.act_fn = nn.GELU()

    def forward(self, x):
        x = self.Linear(x)
        x = self.act_fn(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
