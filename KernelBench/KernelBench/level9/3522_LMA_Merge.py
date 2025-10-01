import torch
from torch import nn


class LMA_Merge(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lamb = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        return x + self.lamb * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
