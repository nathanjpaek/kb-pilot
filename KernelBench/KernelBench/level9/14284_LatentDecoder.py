import torch
from torch import nn


class LatentDecoder(nn.Module):

    def __init__(self, hidden_size):
        super(LatentDecoder, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_mu = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        latent_output = self.dense(hidden_states)
        original_output = self.dense_mu(latent_output)
        original_output = self.LayerNorm(original_output)
        original_output = self.activation(original_output)
        return original_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
