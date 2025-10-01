import random
import torch
import torch.nn as nn


class FLeakyReLU(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        super(FLeakyReLU, self).__init__()
        self.negative_slope = random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.leaky_relu(x, self.negative_slope)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
