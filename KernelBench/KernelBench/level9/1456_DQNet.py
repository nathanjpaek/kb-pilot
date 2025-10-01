import torch
import torch.nn as nn
from torch.nn import functional as F


class DQNet(nn.Module):
    """
    Definition: DQNet(obs_size,act_size,hid_size=256)

    Regular Deep Q Network with three Linear layers
    """

    def __init__(self, obs_size, act_size, hid_size=256):
        super().__init__()
        self.fc_in = nn.Linear(obs_size, hid_size)
        self.fc_out = nn.Linear(hid_size, act_size)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        return self.fc_out(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'act_size': 4}]
