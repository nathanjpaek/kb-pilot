import torch
import torch.nn as nn


def pono(x, epsilon=1e-05):
    """Positional normalization"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output, mean, std


class PONO(nn.Module):

    def forward(self, x, mask=None):
        x, _, __ = pono(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
