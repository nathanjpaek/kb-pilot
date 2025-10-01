import torch
from torch import nn
import torch.utils


class MLP(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.d1 = torch.nn.Linear(input_dim, 32)
        self.d2 = torch.nn.Linear(32, 16)
        self.d3 = torch.nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.d1(x)
        x = self.relu(x)
        x = self.d2(x)
        x = self.relu(x)
        x = self.d3(x)
        x = self.softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
