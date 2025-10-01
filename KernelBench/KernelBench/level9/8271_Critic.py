import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        z = torch.tanh(self.l1(state))
        z = torch.tanh(self.l2(z))
        v = self.l3(z)
        return v


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4}]
