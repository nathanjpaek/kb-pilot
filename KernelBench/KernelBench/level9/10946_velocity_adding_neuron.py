import torch
import torch.nn as nn


class velocity_adding_neuron(nn.Module):

    def __init__(self, weight):
        super(velocity_adding_neuron, self).__init__()
        self.w = weight
        self.nl = nn.Tanh()

    def forward(self, x):
        return self.nl(self.w * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weight': 4}]
