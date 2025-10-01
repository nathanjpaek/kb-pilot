import torch
import torch.nn as nn


class ModelBasic(nn.Module):
    """parallel passing of data, categorical output with one unit per number of clusters"""

    def __init__(self, n_obs, n_units=100, n_timesteps=10, max_K=10):
        super(ModelBasic, self).__init__()
        self.fc_input = nn.Linear(2 * n_obs, n_units)
        self.fc_output = nn.Linear(n_units, max_K)
        self.fc_recurrent = nn.Linear(n_units, n_units)
        self.n_obs = n_obs
        self.n_units = n_units
        self.n_timesteps = n_timesteps
        self.max_K = max_K

    def forward(self, x):
        inp = x.view(-1, 2 * self.n_obs)
        hidden = torch.tanh(self.fc_input(inp))
        for i_time in range(self.n_timesteps):
            hidden = hidden + torch.tanh(self.fc_recurrent(hidden)
                ) + torch.tanh(self.fc_input(inp))
        out = self.fc_output(hidden)
        return out

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_input.weight)
        nn.init.xavier_uniform_(self.fc_output.weight)
        nn.init.xavier_uniform_(self.fc_recurrent.weight)
        self.fc_input.bias.data.fill_(0)
        self.fc_output.bias.data.fill_(0)
        self.fc_recurrent.bias.data.fill_(0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_obs': 4}]
