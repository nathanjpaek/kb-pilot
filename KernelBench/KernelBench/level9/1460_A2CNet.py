import torch
import torch.nn as nn
from torch.nn import functional as F


class A2CNet(nn.Module):
    """Double heads actor + critic network."""

    def __init__(self, obs_size, act_size, hid_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hid_size)
        self.policy = nn.Linear(hid_size, act_size)
        self.value = nn.Linear(hid_size, 1)

    def forward(self, x):
        """Feed forward."""
        y = F.relu(self.fc1(x))
        return self.policy(y), self.value(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'act_size': 4}]
