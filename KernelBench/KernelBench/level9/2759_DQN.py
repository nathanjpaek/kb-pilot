import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, fc1_unit=64, fc2_unit=64,
        fc3_unit=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.A_fc1 = nn.Linear(fc2_unit, fc3_unit)
        self.V_fc1 = nn.Linear(fc2_unit, fc3_unit)
        self.A_fc2 = nn.Linear(fc3_unit, action_dim)
        self.V_fc2 = nn.Linear(fc3_unit, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        A = F.relu(self.A_fc1(x))
        V = F.relu(self.V_fc1(x))
        A = self.A_fc2(A)
        V = self.V_fc2(V)
        Q = V + (A - A.mean())
        return Q


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
