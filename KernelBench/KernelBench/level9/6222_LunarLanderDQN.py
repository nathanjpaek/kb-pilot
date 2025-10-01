import torch
import torch.nn as nn
import torch.nn.functional as F


class LunarLanderDQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim, hidden=12):
        super(LunarLanderDQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_space_dim': 4, 'action_space_dim': 4}]
