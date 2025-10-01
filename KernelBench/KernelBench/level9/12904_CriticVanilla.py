import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBase(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(MLPBase, self).__init__()
        self.l1 = nn.Linear(num_inputs, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class CriticVanilla(nn.Module):
    """a vanilla critic module that outputs a node's q-values given only its observation and action(no message between nodes)"""

    def __init__(self, state_dim, action_dim):
        super(CriticVanilla, self).__init__()
        self.baseQ1 = MLPBase(state_dim + action_dim, 1)
        self.baseQ2 = MLPBase(state_dim + action_dim, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], -1)
        x1 = self.baseQ1(xu)
        x2 = self.baseQ2(xu)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], -1)
        x1 = self.baseQ1(xu)
        return x1


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
