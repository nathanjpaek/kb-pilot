import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, 24)
        self.l5 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l5(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
