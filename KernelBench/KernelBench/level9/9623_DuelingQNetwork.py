import torch
import torch.nn.functional as F
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.afc1 = nn.Linear(state_size, 1024)
        self.afc2 = nn.Linear(1024, 512)
        self.afc3 = nn.Linear(512, action_size)
        self.vfc1 = nn.Linear(state_size, 512)
        self.vfc2 = nn.Linear(512, 512)
        self.vfc3 = nn.Linear(512, action_size)

    def forward(self, state):
        adv = F.relu(self.afc1(state))
        adv = F.relu(self.afc2(adv))
        adv = self.afc3(adv)
        val = F.relu(self.vfc1(state))
        val = F.relu(self.vfc2(val))
        val = self.vfc3(val)
        out = val + (adv - adv.mean())
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
