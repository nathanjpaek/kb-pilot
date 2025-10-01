import random
import torch
import torch.nn as nn


class FPELU(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        super(FPELU, self).__init__()
        self.alpha = random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.elu(x, alpha=self.alpha)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
