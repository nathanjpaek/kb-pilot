import torch
from torch import nn


class Coskx(nn.Module):

    def __init__(self, k=50):
        super(Coskx, self).__init__()
        self.k = k

    def forward(self, input):
        return torch.cos(input * self.k)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
