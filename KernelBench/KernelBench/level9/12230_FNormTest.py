import torch
import torch.nn as nn


class FNormTest(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        super(FNormTest, self).__init__()

    def forward(self, x):
        x = torch.norm(x, p=2, dim=[1, 2])
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
