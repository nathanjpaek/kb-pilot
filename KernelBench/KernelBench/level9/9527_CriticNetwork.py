import torch
import torch.nn.functional as F
import torch.nn as nn


class CriticNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(CriticNetwork, self).__init__()
        torch.manual_seed(seed)
        fcs1_units = 64
        fc2_units = 64
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
