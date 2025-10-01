import torch
import torch.utils.data
from torch import nn
from torch.nn import functional


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim):
        """
        Args:
                input_dim: A integer indicating the size of input.
                hidden_dim: A integer indicating the size of hidden dimension.
                z_dim: A integer indicating the latent dimension.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = functional.relu(self.linear(x))
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)
        return z_mu, z_var


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'z_dim': 4}]
