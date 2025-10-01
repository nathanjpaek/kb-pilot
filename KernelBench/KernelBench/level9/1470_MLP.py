import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron network
    """

    def __init__(self, obs_dim, dim_latent):
        """
        Constructor

        Args:
            obs_dim: (int) dimension of observation
            latent_dim: (int) dimension of output latent
        """
        super().__init__()
        self._hidden_dim = 20
        self.fc1 = nn.Linear(obs_dim, self._hidden_dim)
        self.fc2 = nn.Linear(self._hidden_dim, dim_latent)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        forward method
        """
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_dim': 4, 'dim_latent': 4}]
