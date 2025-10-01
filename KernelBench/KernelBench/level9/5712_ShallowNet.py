import torch
import torch.nn as nn


class ShallowNet(nn.Module):

    def __init__(self, n_features):
        super(ShallowNet, self).__init__()
        self.a1 = nn.Linear(n_features, 2)

    def forward(self, x):
        return torch.sigmoid(self.a1(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
