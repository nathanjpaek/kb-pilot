import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteCriticNetwork(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super(DiscreteCriticNetwork, self).__init__()
        self._l1 = nn.Linear(obs_dim, hidden_size)
        self._l2 = nn.Linear(hidden_size, hidden_size)
        self._l3 = nn.Linear(hidden_size, act_dim)

    def forward(self, s, a):
        s = F.relu(self._l1(s))
        s = F.relu(self._l2(s))
        s = self._l3(s)
        return s.gather(1, a.long())


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_dim': 4, 'act_dim': 4}]
