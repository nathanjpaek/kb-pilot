import torch
import torch.nn as nn


class FiLM(nn.Module):

    def __init__(self, zdim, maskdim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(zdim, maskdim)
        self.beta = nn.Linear(zdim, maskdim)

    def forward(self, x, z):
        gamma = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'zdim': 4, 'maskdim': 4}]
