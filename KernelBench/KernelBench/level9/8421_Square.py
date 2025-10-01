import torch
import torch.nn as nn


class Square(nn.Module):

    def __init__(self):
        super(Square, self).__init__()

    def forward(self, X):
        return torch.square(X)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
