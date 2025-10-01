import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, observations_dim, actions_dim, hidden_dim=500):
        super(Net, self).__init__()
        self._input_layer = nn.Linear(observations_dim, hidden_dim)
        self._hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self._output_layer = nn.Linear(hidden_dim, actions_dim)

    def forward(self, x):
        x = F.relu(self._input_layer(x))
        x = F.relu(self._hidden1(x))
        x = self._output_layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'observations_dim': 4, 'actions_dim': 4}]
