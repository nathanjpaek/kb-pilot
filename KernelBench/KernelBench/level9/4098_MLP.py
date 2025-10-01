import torch
import torch as th
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden=128):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden, bias=False)
        self.linear2 = nn.Linear(hidden, output_size, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = th.tanh(x)
        x = self.linear2(x)
        x = th.tanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
