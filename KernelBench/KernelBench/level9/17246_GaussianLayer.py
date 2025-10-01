import torch
import torch.nn as nn


class GaussianLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GaussianLayer, self).__init__()
        self.z_mu = torch.nn.Linear(input_dim, output_dim)
        self.z_sigma = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = self.z_mu(x)
        std = self.z_sigma(x)
        std = torch.exp(std)
        return mu, std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
