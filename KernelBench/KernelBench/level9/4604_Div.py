import torch
import torch.nn as nn


class Div(nn.Module):

    def __init__(self):
        super(Div, self).__init__()

    def forward(self, x):
        x = torch.div(x, 0.5)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
