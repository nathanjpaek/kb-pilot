import torch
import torch.nn as nn
import torch.nn.functional as F


class CartpoleDQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim, hidden=12):
        super(CartpoleDQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_space_dim': 4, 'action_space_dim': 4}]
