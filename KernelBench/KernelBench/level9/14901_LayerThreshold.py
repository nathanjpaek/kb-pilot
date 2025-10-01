import random
import torch
import torch.nn as nn


class LayerThreshold(nn.Module):
    """
    Test for nn.layers based types
    """

    def __init__(self):
        super(LayerThreshold, self).__init__()
        self.threshold = random.random()
        self.value = self.threshold + random.random()
        self.thresh = nn.Threshold(self.threshold, self.value)

    def forward(self, x):
        x = self.thresh(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
