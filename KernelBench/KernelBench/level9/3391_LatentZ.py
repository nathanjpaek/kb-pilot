import torch
import torch.nn as nn


class LatentZ(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, p_x):
        mu = self.mu(p_x)
        logvar = self.logvar(p_x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return std * eps + mu, logvar, mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'latent_size': 4}]
