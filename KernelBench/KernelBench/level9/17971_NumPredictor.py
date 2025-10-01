import torch
import torch.nn.functional as F
import torch.nn as nn


class NumPredictor(nn.Module):

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(NumPredictor, self).__init__()
        self.reg_1 = nn.Linear(self.latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.reg_1(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'latent_dim': 4}]
