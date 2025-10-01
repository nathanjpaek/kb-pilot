import torch
import torch.nn as nn


class BiasLayer(nn.Module):

    def __init__(self, channels, skip_dims=2):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels, *([1] * skip_dims)))

    def forward(self, net):
        return net + self.bias

    def extra_repr(self):
        return f'shape={self.bias.shape}'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
