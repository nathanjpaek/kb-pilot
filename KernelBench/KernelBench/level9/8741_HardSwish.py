import torch
import torch.nn.functional as F
from torch import nn


class HardSwish(nn.Module):

    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, True) / 6.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
