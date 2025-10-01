import torch
import torch.nn as nn


class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x):
        x = torch.add(x, 20)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
