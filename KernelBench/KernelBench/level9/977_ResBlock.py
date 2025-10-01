import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer1 = nn.Linear(self.size, self.size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(self.size, self.size)

    def forward(self, x):
        shortcut = x
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x + shortcut
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
