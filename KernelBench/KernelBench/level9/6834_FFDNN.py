import torch
import torch as tc
import torch.nn as nn


class FFDNN(nn.Module):

    def __init__(self, insize, action_space):
        super(FFDNN, self).__init__()
        self.input = nn.Linear(insize, 64)
        self.layer1 = nn.Linear(64, 32)
        self.layer2 = nn.Linear(32, action_space)

    def forward(self, x):
        x = tc.tanh(self.input(x))
        x = tc.tanh(self.layer1(x))
        x = self.layer2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'insize': 4, 'action_space': 4}]
