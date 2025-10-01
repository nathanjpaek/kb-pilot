import torch
import torch.nn as nn


class LinearActor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(LinearActor, self).__init__()
        self.l1 = nn.Linear(state_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * torch.sigmoid(self.l1(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'max_action': 4}]
