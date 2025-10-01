import torch
import torch.nn as nn
import torch.nn.functional as F


class SortNet(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SortNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size, bias=None)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
