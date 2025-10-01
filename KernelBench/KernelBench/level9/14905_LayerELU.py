import random
import torch
import torch.nn as nn


class LayerELU(nn.Module):
    """
    Test for nn.layers based types
    """

    def __init__(self):
        super(LayerELU, self).__init__()
        self.alpha = random.random()
        self.elu = nn.ELU(alpha=self.alpha)

    def forward(self, x):
        x = self.elu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
