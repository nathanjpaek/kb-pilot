import torch
from torch import nn


class LReluCustom(nn.Module):

    def __init__(self, leak=0.1):
        super(LReluCustom, self).__init__()
        self.leak = leak

    def forward(self, x):
        return torch.max(x, self.leak * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
