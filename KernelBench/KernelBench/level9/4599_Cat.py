import torch
import torch.nn as nn


class Cat(nn.Module):

    def __init__(self):
        super(Cat, self).__init__()

    def forward(self, x):
        addition = torch.split(x, 2, dim=1)[0]
        None
        x = torch.cat([x, addition], dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
