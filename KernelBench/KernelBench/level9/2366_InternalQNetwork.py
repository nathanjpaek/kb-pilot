import torch
import torch.nn.functional as F
import torch.nn as nn


class InternalQNetwork(nn.Module):

    def __init__(self, state_size, action_size, recurrent_size, seed,
        fc1_units=64, fc2_units=128):
        super(InternalQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + recurrent_size, fc2_units)
        self.fc3 = nn.Linear(fc1_units + fc2_units, recurrent_size)
        self.fc4 = nn.Linear(fc2_units, action_size)

    def forward(self, x):
        obs = x[:, :8]
        prev_recurrent = x[:, -5:]
        x1 = F.relu(self.fc1(obs))
        x2 = F.relu(self.fc2(torch.cat([x1, prev_recurrent], dim=1)))
        recurrent_activation = torch.sigmoid(self.fc3(torch.cat([x1, x2],
            dim=1)))
        action_activation = self.fc4(x2)
        return torch.cat([action_activation, recurrent_activation], dim=1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'recurrent_size': 4,
        'seed': 4}]
