import random
import torch
import torch.nn as nn


class LayerLeakyReLU(nn.Module):
    """
    Test for nn.layers based types
    """

    def __init__(self):
        super(LayerLeakyReLU, self).__init__()
        self.negative_slope = random.random()
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, x):
        x = self.leaky_relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
