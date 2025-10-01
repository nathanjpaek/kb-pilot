import torch
import torch.nn as nn


class PermEqMean(nn.Module):
    """ Returns equivariant layer used by EquivarDrift. """

    def __init__(self, in_dim, out_dim):
        super(PermEqMean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x_mean = x.mean(-2, keepdim=True)
        return self.Gamma(x) + self.Lambda(x_mean)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
