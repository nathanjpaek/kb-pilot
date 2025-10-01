import random
import torch
import torch.nn as nn


class LayerHardtanh(nn.Module):
    """
    Test for nn.layers based types
    """

    def __init__(self):
        super(LayerHardtanh, self).__init__()
        self.min_val = random.random()
        self.max_val = self.min_val + random.random()
        self.htanh = nn.Hardtanh(min_val=self.min_val, max_val=self.max_val)

    def forward(self, x):
        x = self.htanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
