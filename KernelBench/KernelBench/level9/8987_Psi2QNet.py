import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


class Psi2QNet(nn.Module):

    def __init__(self, output_dim, feature_dim):
        super(Psi2QNet, self).__init__()
        self.w = Parameter(torch.Tensor(feature_dim))
        nn.init.constant_(self.w, 0)
        self

    def forward(self, psi):
        return torch.matmul(psi, self.w)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'output_dim': 4, 'feature_dim': 4}]
