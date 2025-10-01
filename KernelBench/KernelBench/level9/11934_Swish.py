import torch
from torch.functional import F


class Swish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.sigmoid(x) * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
