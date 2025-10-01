import torch
import torch.nn as nn


class RelevanceVector(nn.Module):

    def __init__(self, z_dim):
        super(RelevanceVector, self).__init__()
        self.rvlogit = nn.Parameter(0.001 * torch.randn(z_dim))

    def forward(self):
        rv = torch.sigmoid(self.rvlogit)
        return self.rvlogit, rv


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'z_dim': 4}]
