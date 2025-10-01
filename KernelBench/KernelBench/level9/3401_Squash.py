import torch
import torch.nn as nn
import torch.jit


class Squash(nn.Module):

    def forward(self, x):
        y = x ** 3
        return torch.clamp(y, min=0) / (1 + y.abs())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
