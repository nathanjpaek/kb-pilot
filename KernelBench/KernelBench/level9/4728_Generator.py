import torch
from torch import nn
import torch.nn.functional as f


class Generator(nn.Module):

    def __init__(self, nz):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(nz, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nz': 4}]
