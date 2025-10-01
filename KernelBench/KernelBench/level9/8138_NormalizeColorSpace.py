import torch
from torch import nn
from typing import *


class NormalizeColorSpace(nn.Module):

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x.clamp(0.0, 255.0)
        return x / 255.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
