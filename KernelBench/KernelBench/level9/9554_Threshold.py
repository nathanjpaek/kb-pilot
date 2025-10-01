import torch
from torch import nn


class Threshold(nn.Module):

    def __init__(self, threshold):
        super(Threshold, self).__init__()
        self.threshold = nn.Threshold(threshold, 0.0)

    def forward(self, x):
        return self.threshold(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'threshold': 4}]
