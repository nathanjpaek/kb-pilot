import torch
import numpy as np
import torch.nn as nn


class FSoftmaxTest(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        super(FSoftmaxTest, self).__init__()
        self.dim = np.random.randint(0, 3)

    def forward(self, x):
        from torch.nn import functional as F
        return F.softmax(x, self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
