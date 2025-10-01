import torch
import torch.nn as nn


class PrefModel(nn.Module):

    def __init__(self, input_dim):
        super(PrefModel, self).__init__()
        self.combination = nn.Linear(input_dim, 2)
        self.softmax = nn.Softmax(1)

    def forward(self, features):
        h = self.combination(features)
        out = self.softmax(h)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
