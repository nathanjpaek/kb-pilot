import torch
import torch.nn as nn


class PixelwiseNormalization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        factor = ((x ** 2).mean(dim=1, keepdim=True) + 1e-08) ** 0.5
        return x / factor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
