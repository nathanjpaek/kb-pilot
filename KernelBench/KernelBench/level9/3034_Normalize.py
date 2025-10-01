import torch
import torch.nn as nn


class Normalize(nn.Module):

    def forward(self, x):
        return (x - 0.1307) / 0.3081


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
