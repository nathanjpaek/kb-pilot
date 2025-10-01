import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorDeep(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(ActorDeep, self).__init__()
        self.l1 = nn.Linear(state_dim, 300)
        self.l2 = nn.Linear(300, 300)
        self.l3 = nn.Linear(300, 300)
        self.l4 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.max_action * torch.tanh(self.l4(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'max_action': 4}]
