import torch
import torch.nn as nn
from torch.nn import functional as F


class BigRamDuel(nn.Module):
    """
    Definition: DuelDQNet(obs_size, act_size)
    """

    def __init__(self, obs_size, act_size):
        super().__init__()
        self.base = nn.Linear(obs_size, 256)
        self.fc1 = nn.Linear(256, 256)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(128, 64)
        self.val = nn.Linear(64, 1)
        self.adv = nn.Linear(64, act_size)

    def forward(self, x):
        x /= 255
        out = F.relu(self.base(x))
        out = F.relu(self.fc1(out))
        out = self.drop1(out)
        out = F.relu(self.fc2(out))
        out = self.drop2(out)
        out = F.relu(self.fc3(out))
        val = self.val(out)
        adv = self.adv(out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'act_size': 4}]
