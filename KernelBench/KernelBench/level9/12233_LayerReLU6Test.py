import torch
import torch.nn as nn


class LayerReLU6Test(nn.Module):
    """
    Test for nn.layers based types
    """

    def __init__(self):
        super(LayerReLU6Test, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
