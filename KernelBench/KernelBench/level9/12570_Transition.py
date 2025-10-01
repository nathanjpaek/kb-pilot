import torch
import torch.nn as nn


class Transition(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super(Transition, self).__init__()
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_loc = nn.Linear(hidden_dim, z_dim)
        self.hidden_to_sig = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        hidden1 = self.relu(self.z_to_hidden(z_t_1))
        hidden2 = self.relu(self.hidden_to_hidden(hidden1))
        loc = self.hidden_to_loc(hidden2)
        sigma = self.softplus(self.hidden_to_sig(hidden2))
        return loc, sigma


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4, 'hidden_dim': 4}]
