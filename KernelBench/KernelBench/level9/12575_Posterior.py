import torch
import torch.nn as nn


class Posterior(nn.Module):

    def __init__(self, z_dim, hidden_dim, obs_dim):
        super(Posterior, self).__init__()
        self.z_obs_to_hidden = nn.Linear(2 * z_dim + obs_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_loc = nn.Linear(hidden_dim, z_dim)
        self.hidden_to_sig = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_mu, z_sig, obs_t):
        hidden1 = self.relu(self.z_obs_to_hidden(torch.cat((z_mu, z_sig,
            obs_t), dim=-1)))
        hidden2 = self.relu(self.hidden_to_hidden(hidden1))
        loc = self.hidden_to_loc(hidden2)
        sig = self.softplus(self.hidden_to_sig(hidden2))
        return loc, sig


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4, 'hidden_dim': 4, 'obs_dim': 4}]
