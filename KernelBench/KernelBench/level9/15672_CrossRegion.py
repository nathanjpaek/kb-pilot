import torch
import torch.nn as nn
import torch.fft


class CrossRegion(nn.Module):

    def __init__(self, step=1, dim=1):
        super().__init__()
        self.step = step
        self.dim = dim

    def forward(self, x):
        return torch.roll(x, self.step, self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
