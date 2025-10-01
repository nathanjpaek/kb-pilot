import torch
import torch.nn as nn


class FocusLayer(nn.Module):

    def __init__(self, c1, c2, k=1):
        super(FocusLayer, self).__init__()

    def forward(self, x):
        return torch.cat([x[..., ::2], x[..., 1::2]], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c1': 4, 'c2': 4}]
