import torch
from torch import nn


class SimpleNN(nn.Module):

    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, 50)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(50, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
