import torch
import torch.nn as nn
import torch.utils.cpp_extension


class PixelNorm(nn.Module):
    """pixel normalization"""

    def forward(self, x):
        x = x / x.pow(2).mean(dim=1, keepdim=True).sqrt().add(1e-08)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
