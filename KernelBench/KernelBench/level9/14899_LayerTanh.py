import torch
import torch.nn as nn


class LayerTanh(nn.Module):
    """
    Test for nn.layers based types
    """

    def __init__(self):
        super(LayerTanh, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
