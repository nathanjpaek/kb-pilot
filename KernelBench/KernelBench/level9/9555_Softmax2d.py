import torch
from torch import nn


class Softmax2d(nn.Module):

    def __init__(self):
        super().__init__()
        self.Softmax2d = nn.Softmax2d()

    def forward(self, x):
        x = self.Softmax2d(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
