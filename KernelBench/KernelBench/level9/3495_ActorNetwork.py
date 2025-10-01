import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):

    def __init__(self, obs_dim, hidden_size=256):
        super(ActorNetwork, self).__init__()
        self._obs_dim = obs_dim
        self._l1 = nn.Linear(obs_dim, hidden_size)
        self._l2 = nn.Linear(hidden_size, hidden_size)
        self._hidden_size = hidden_size

    def forward(self, x):
        x = F.relu(self._l1(x))
        x = F.relu(self._l2(x))
        return x

    def policy(self, x):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_dim': 4}]
