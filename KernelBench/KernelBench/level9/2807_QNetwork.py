import torch
import torch.nn.functional as F
import torch.nn as nn


class QNetwork(nn.Module):
    """ Actor Policy (Q Network) model """

    def __init__(self, state_size, action_size, seed, fc1_units=512,
        fc2_units=256, fc3_units=64):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            fcx_units (int): Dimension of hidden sizes, x = ith layer
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, fc2_units)
        self.fc4 = nn.Linear(fc2_units, fc3_units)
        self.fc5 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        action = self.fc5(x)
        return action


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
