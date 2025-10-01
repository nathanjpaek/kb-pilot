import torch
from torch import nn


class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        """
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        """
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

    def forward(self, h):
        _, _hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-06
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'output_size': 4}]
