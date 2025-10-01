import torch
import torch.nn as nn


class CReLU(nn.Module):

    def __init__(self):
        super(CReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.cat((self.relu(x), self.relu(-x)), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
