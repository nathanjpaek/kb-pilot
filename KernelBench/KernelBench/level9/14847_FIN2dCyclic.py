import torch
import torch.utils.data
import torch
import torch.nn as nn


class FIN2dCyclic(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(dim, affine=False)
        self.a_gamma = nn.Parameter(torch.zeros(dim))
        self.b_gamma = nn.Parameter(torch.ones(dim))
        self.a_beta = nn.Parameter(torch.zeros(dim))
        self.b_beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x, cos, sin):
        gamma = self.a_gamma * cos.unsqueeze(-1) + self.b_gamma
        beta = self.a_beta * sin.unsqueeze(-1) + self.b_beta
        return self.instance_norm(x) * gamma.unsqueeze(-1).unsqueeze(-1
            ) + beta.unsqueeze(-1).unsqueeze(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
