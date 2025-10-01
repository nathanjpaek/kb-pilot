import torch
import numpy as np
import torch.nn as nn


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.forward1 = nn.Linear(s_dim, 400)
        self.Relu = nn.ReLU()
        self.forward2 = nn.Linear(400, 300)
        self.forward3 = nn.Linear(300, a_dim)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = fanin_init(m.weight.data.size())
        self.forward2.weight.data.uniform_(-0.003, 0.003)

    def forward(self, x):
        x = self.forward1(x)
        x = self.tanh(x)
        x = self.forward2(x)
        x = self.Relu(x)
        x = self.forward3(x)
        x = self.tanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'s_dim': 4, 'a_dim': 4}]
