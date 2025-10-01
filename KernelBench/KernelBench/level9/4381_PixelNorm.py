import torch
import torch.nn as nn
import torch.utils.cpp_extension
import torch.utils.data.distributed


class PixelNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=
            True) + 1e-08)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
