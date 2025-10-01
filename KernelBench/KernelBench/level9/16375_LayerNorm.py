import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1).expand_as(x)
        std = x.std(-1).expand_as(x)
        return self.gamma.expand_as(x) * (x - mean) / (std + self.eps
            ) + self.beta.expand_as(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
