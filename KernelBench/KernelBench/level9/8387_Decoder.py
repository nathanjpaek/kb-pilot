import torch
import torch.utils.data
from torch import nn
from torch.nn import functional


class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_dim, output_dim):
        """
        Args:
                z_dim: A integer indicating the latent size.
                hidden_dim: A integer indicating the size of hidden dimension.
                output_dim: A integer indicating the output dimension.
        """
        super().__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = functional.relu(self.linear(x))
        predicted = torch.sigmoid(self.out(hidden))
        return predicted


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4, 'hidden_dim': 4, 'output_dim': 4}]
