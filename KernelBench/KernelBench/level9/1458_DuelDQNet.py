import torch
import torch.nn as nn
from torch.nn import functional as F


class DuelDQNet(nn.Module):
    """
    Definition: DuelDQNet(obs_size, act_size, hid_size=256)
    """

    def __init__(self, obs_size, act_size, hid_size=256):
        super().__init__()
        self.base = nn.Linear(obs_size, hid_size)
        self.val = nn.Linear(hid_size, 1)
        self.adv = nn.Linear(hid_size, act_size)

    def forward(self, x):
        x = F.relu(self.base(x))
        val = self.val(x)
        adv = self.adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'act_size': 4}]
