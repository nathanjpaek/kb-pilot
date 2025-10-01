import torch
import torch.nn as nn
from torch.autograd import *
import torch.nn.init as init


class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=0.001):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(
            ln_out)
        return ln_out


class LinearEncoder(nn.Module):
    """docstring for LinearEncoder"""

    def __init__(self, d_model, d_inner, dropout=0.1):
        super(LinearEncoder, self).__init__()
        self.proj1 = nn.Linear(d_model, d_inner)
        self.layer_norm = LayerNormalization(d_inner)
        init.xavier_normal_(self.proj1.weight)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.proj1(x))
        output = self.dropout(output)
        return self.layer_norm(output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_inner': 4}]
