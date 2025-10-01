import torch
from typing import Tuple
from torch import nn


class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.
    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.
    z_dim : int
        Dimension of latent variable z.
    """

    def __init__(self, r_dim: 'int', z_dim: 'int') ->None:
        super(MuSigmaEncoder, self).__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = torch.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'r_dim': 4, 'z_dim': 4}]
