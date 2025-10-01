import torch
import torch.nn as nn


class BertLayerNormNoVar(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNormNoVar, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = x - u
        return self.weight * x + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
