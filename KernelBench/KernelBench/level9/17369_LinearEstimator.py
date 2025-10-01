import torch
import torch.nn as nn


class LinearEstimator(nn.Module):

    def __init__(self, in_c, out_c, factor=1.0):
        super().__init__()
        self.factor = factor
        self.net = nn.Linear(in_c, out_c, bias=False)

    def forward(self, x):
        return self.net(x) * self.factor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_c': 4, 'out_c': 4}]
