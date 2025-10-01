import torch
import torch.nn as nn
import torch.hub


class AddTensors(nn.Module):
    """ Adds all its inputs together. """

    def forward(self, xs):
        return sum(xs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
