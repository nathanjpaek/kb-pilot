import torch
import torch.nn as nn


class QNet(nn.Module):

    def __init__(self, in_size: 'int', out_size: 'int'):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(in_size, 16)
        self.fc_out = nn.Linear(16, out_size)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        o1 = self.act(self.fc1(x))
        o2 = self.act(self.fc_out(o1))
        return o2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
