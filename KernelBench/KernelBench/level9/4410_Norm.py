import torch
from torch import nn


class Norm(nn.Module):

    def __init__(self, order=1, size_average=True):
        super().__init__()
        self.order = order
        self.average = size_average

    def forward(self, inp, target=None):
        if target is not None:
            inp = inp - target
        inp = inp.flatten()
        norm = torch.norm(inp, p=self.order)
        if self.average:
            norm = norm / len(inp)
        return norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
