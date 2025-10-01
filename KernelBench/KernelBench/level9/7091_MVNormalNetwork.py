import torch
import torch.nn as nn


class MVNormalNetwork(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.sc = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        mean = self.mean(x)
        sc = self.sc(x)
        return mean, torch.diag_embed(torch.exp(sc))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'latent_dim': 4}]
