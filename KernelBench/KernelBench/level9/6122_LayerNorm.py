import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.nn import Parameter
from torch.nn.parameter import Parameter


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=0.0001):
        super(LayerNorm, self).__init__()
        self.alpha = Parameter(torch.ones(1, 1, hidden_size))
        self.beta = Parameter(torch.zeros(1, 1, hidden_size))
        self.eps = eps

    def forward(self, x):
        mu = torch.mean(x, 2, keepdim=True).expand_as(x)
        sigma = torch.std(x, 2, keepdim=True).expand_as(x)
        return (x - mu) / (sigma + self.eps) * self.alpha.expand_as(x
            ) + self.beta.expand_as(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
