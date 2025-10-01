import torch
import torch.nn as nn


class FFloorTest(nn.Module):
    """
    Test for nn.functional types
    """

    def __init__(self):
        super(FFloorTest, self).__init__()

    def forward(self, x):
        return x.floor()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
