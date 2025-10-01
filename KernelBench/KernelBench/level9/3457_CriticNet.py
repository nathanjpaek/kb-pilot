from torch.nn import Module
import torch
from torch.nn import Linear
import torch.nn.functional as F


class CriticNet(Module):

    def __init__(self, hidden_size):
        super(CriticNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size // 2)
        self.l2 = Linear(hidden_size // 2, 1)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = torch.tanh(self.l2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
