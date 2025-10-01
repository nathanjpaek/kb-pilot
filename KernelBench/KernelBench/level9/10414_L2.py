import torch
import torch.nn as nn


class L2(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, s, t):
        out = (s - t) ** 2
        return (out.view(out.size(0), -1).sum(dim=1) + 1e-14) ** 0.5


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
