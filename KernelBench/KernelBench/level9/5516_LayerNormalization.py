import torch
import torch.nn as nn


class LayerNormalization(nn.Module):

    def __init__(self, hidden_size, eps=1e-05):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z)
        sigma = torch.std(z)
        ln_out = (z - mu) / (sigma + self.eps)
        ln_out = ln_out * self.a2 + self.b2
        return ln_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
