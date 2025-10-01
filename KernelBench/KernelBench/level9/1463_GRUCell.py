import torch
import torch.nn as nn
from torch.nn import functional as F


class GRUCell(nn.Module):
    """Plain vanilla policy gradient network."""

    def __init__(self, obs_size, act_size, hid_size=128):
        super().__init__()
        self.fc1 = nn.GRUCell(obs_size, hid_size)
        self.output = nn.Linear(hid_size, act_size, bias=True)

    def forward(self, obs):
        """Feed forward."""
        output = F.relu(self.fc1(obs))
        return self.output(output)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'act_size': 4}]
