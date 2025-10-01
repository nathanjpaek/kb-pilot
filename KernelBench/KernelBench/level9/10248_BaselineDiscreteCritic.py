import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import F
from torch.nn import functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributions


class BaselineDiscreteCritic(nn.Module):

    def __init__(self, obs_shape, action_shape, hidden_size=300):
        super().__init__()
        self.fc1 = nn.Linear(obs_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_shape)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        vals = self.out(x)
        return vals


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_shape': 4, 'action_shape': 4}]
