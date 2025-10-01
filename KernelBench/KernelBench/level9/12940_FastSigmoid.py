import torch
import torch.utils.data
import torch
import torch.nn as nn


class FastSigmoid(nn.Module):

    def __init__(self):
        super(FastSigmoid, self).__init__()

    def forward(self, x):
        abs = torch.abs(x) + 1
        return torch.div(x, abs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
