import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data


class Transformer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Transformer, self).__init__()
        self.T_sigma = nn.Linear(in_channels, out_channels)
        self.T_gamma = nn.Linear(in_channels, out_channels)

    def forward(self, sigma, gamma):
        sigma_out = self.T_sigma(sigma)
        gamma_out = self.T_gamma(gamma)
        return F.softplus(sigma_out), F.softplus(gamma_out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
