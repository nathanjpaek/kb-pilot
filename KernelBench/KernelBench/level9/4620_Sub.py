import torch
import torch.nn as nn


class Sub(nn.Module):

    def __init__(self):
        super(Sub, self).__init__()

    def forward(self, x):
        x = torch.sub(x, 20)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
