import torch
import numpy as np
import torch.nn as nn


class GaussianGenerator(nn.Module):

    def __init__(self, dims):
        super(GaussianGenerator, self).__init__()
        self.z_dim = dims[0]
        self.linear_var = nn.Parameter(1.0 * torch.ones([self.z_dim]))
        self.bias = nn.Parameter(torch.zeros([self.z_dim]))
        self.lmbda = 0.001
        self.dist = None

    def forward(self, z):
        out = z * self.linear_var ** 2
        out = out + self.lmbda * z + self.bias
        return out

    def log_density(self, x):
        Sigma = self.linear_var ** 2 + self.lmbda
        Sigma = Sigma ** 2
        location = x - self.bias
        quad = torch.einsum('nd,nd,d->n', location, location, 1.0 / Sigma)
        quad = quad.unsqueeze(-1)
        value = -0.5 * quad - 0.5 * torch.log(2.0 * np.pi * Sigma).sum()
        return value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dims': [4, 4]}]
