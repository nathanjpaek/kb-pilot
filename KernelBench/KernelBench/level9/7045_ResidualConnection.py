import torch
import torch.nn as nn


class ResidualConnection(nn.Module):

    def __init__(self, *layers):
        super(ResidualConnection, self).__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return (input + self.layers(input)) / 2.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
