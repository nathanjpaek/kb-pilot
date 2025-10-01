import torch
from torch import nn
import torch.nn.functional as F


class Highway(nn.Module):

    def __init__(self, size):
        super(Highway, self).__init__()
        self.one = nn.Linear(size, size)
        self.two = nn.Linear(size, size)

    def forward(self, x):
        x0 = F.relu(self.one(x))
        x1 = torch.sigmoid(self.two(x))
        return x0 * x1 + x * (1.0 - x1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
