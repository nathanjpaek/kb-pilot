import random
import torch
import torch.nn as nn


class FHardtanh(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        super(FHardtanh, self).__init__()
        self.min_val = random.random()
        self.max_val = self.min_val + random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
