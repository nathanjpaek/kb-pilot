import torch
import numpy as np
import torch.nn as nn


class FClipTest(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        self.low = np.random.uniform(-1, 1)
        self.high = np.random.uniform(1, 2)
        super(FClipTest, self).__init__()

    def forward(self, x):
        return x.clamp(self.low, self.high)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
