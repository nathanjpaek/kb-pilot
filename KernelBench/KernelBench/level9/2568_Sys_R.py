import torch
import torch.nn as nn
import torch.nn.functional as F


class Sys_R(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256):
        super(Sys_R, self).__init__()
        self.l1 = nn.Linear(2 * state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)

    def forward(self, state, next_state, action):
        xa = torch.cat([state, next_state, action], 1)
        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
