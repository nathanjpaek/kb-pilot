import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Takes in data, returns mu and sigma for variational approximation of latent variable.
    """

    def __init__(self, alph_size, seq_len, z_dim=30, hidden_architecture=[
        1500, 1500]):
        super(Encoder, self).__init__()
        self.hidden1 = nn.Linear(alph_size * seq_len, hidden_architecture[0])
        self.hidden2 = nn.Linear(hidden_architecture[0], hidden_architecture[1]
            )
        self.final1 = nn.Linear(hidden_architecture[1], z_dim)
        self.final2 = nn.Linear(hidden_architecture[1], z_dim)
        self.relu = nn.ReLU()
        self.alph_size = alph_size
        self.seq_len = seq_len

    def forward(self, x):
        x = x.reshape(-1, self.seq_len * self.alph_size)
        hidden1 = self.relu(self.hidden1(x))
        hidden2 = self.relu(self.hidden2(hidden1))
        z_loc = self.final1(hidden2)
        z_scale = torch.exp(self.final2(hidden2))
        return z_loc, z_scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alph_size': 4, 'seq_len': 4}]
