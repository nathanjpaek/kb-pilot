import torch
from torch import nn


def sigmoid_range(x, low, high):
    """Sigmoid function with range `(low, high)`"""
    return torch.sigmoid(x) * (high - low) + low


class SigmoidRange(nn.Module):
    """Sigmoid module with range `(low,x_max)`"""

    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'low': 4, 'high': 4}]
