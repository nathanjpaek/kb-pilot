import math
import torch
import torch.nn as nn


def gelu(x):
    """Implementation of the gelu activation function by Hugging Face"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, n_hidden):
        super().__init__()
        self.fc1 = nn.Conv2d(n_hidden, n_hidden * 4, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(n_hidden * 4, n_hidden, kernel_size=1, bias=True)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_hidden': 4}]
