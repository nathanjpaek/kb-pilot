import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, 32)
        self.linear2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.view((-1, self.input_size))
        out = F.relu(self.linear1(x))
        out = F.softmax(self.linear2(out), dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
