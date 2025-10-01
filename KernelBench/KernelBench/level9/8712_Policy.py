import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 125)
        self.fc_norm = nn.LayerNorm(125)
        self.fc2 = nn.Linear(125, 125)
        self.fc2_norm = nn.LayerNorm(125)
        self.action_prob = nn.Linear(125, action_size)

    def forward(self, x):
        x = F.relu(self.fc_norm(self.fc1(x)))
        x = F.relu(self.fc2_norm(self.fc2(x)))
        x = F.softmax(self.action_prob(x), dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
