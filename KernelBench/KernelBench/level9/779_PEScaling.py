from torch.nn import Module
import torch
from torch import nn


class PEScaling(Module):

    def __init__(self):
        super(PEScaling, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x):
        E = x.mean(-1).mean(-1).mean(-1).unsqueeze(-1)
        return self.sigmoid(self.linear2(self.relu(self.linear1(E)))
            ).unsqueeze(-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
