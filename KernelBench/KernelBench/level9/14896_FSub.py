import torch
import torch.nn as nn


class FSub(nn.Module):

    def __init__(self):
        super(FSub, self).__init__()

    def forward(self, x, y):
        x = x - y - 8.3
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
