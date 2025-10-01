import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):

    def __init__(self, state_dim, actions_dim, hidden_dim=64):
        super(PolicyNet, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, actions_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        return self.hidden(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'actions_dim': 4}]
