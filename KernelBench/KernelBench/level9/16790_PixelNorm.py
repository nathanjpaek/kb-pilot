import torch
from torch import nn


class PixelNorm(nn.Module):

    def __init__(self, pixel_norm_op_dim):
        super().__init__()
        self.pixel_norm_op_dim = pixel_norm_op_dim

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=self.
            pixel_norm_op_dim, keepdim=True) + 1e-08)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pixel_norm_op_dim': 4}]
