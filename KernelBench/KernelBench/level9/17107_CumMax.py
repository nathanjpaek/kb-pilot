import torch
import torch.nn as nn


class CumMax(nn.Module):

    def __init__(self):
        super(CumMax, self).__init__()

    def forward(self, input):
        return torch.cumsum(nn.Softmax(dim=-1)(input), -1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
