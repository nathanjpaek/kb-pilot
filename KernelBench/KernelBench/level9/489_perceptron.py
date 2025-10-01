import torch
from torch import nn
import torch.nn.functional as F


class perceptron(nn.Module):

    def __init__(self, n_channels):
        super(perceptron, self).__init__()
        self.L = nn.Linear(n_channels, 10)

    def forward(self, x):
        x = self.L(x)
        x = F.softmax(x, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
