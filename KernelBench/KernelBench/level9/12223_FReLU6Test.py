import torch
import torch.nn as nn


class FReLU6Test(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        super(FReLU6Test, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.relu6(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
