import math
import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_parameters_util(model):
    pass


class RewardEstimator(nn.Module):
    """Estimates the reward the agent will receieved. Value used as a baseline in REINFORCE loss"""

    def __init__(self, hid_dim):
        super(RewardEstimator, self).__init__()
        self.hid_dim = hid_dim
        self.v1 = nn.Linear(hid_dim, math.ceil(hid_dim / 2))
        self.v2 = nn.Linear(math.ceil(hid_dim / 2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util(self)

    def forward(self, x):
        x = x.detach()
        x = F.relu(self.v1(x))
        x = self.v2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hid_dim': 4}]
