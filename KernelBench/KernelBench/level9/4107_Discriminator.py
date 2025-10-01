import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 500)
        self.l2 = nn.Linear(500, 300)
        self.l3 = nn.Linear(300, 300)
        self.l4 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
