import torch
import torch.nn as nn


class ScaleLayer(nn.Module):

    def __init__(self, init_value=0.001):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        None
        return input * self.scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
