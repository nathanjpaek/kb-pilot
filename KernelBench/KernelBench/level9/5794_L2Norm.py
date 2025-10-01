import torch
import torch.nn as nn


class L2Norm(nn.Module):
    """Channel-wise L2 normalization."""

    def __init__(self, in_channels):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels))

    def forward(self, x):
        """out = weight * x / sqrt(\\sum x_i^2)"""
        unsqueezed_weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return unsqueezed_weight * x * x.pow(2).sum(1, keepdim=True).clamp(min
            =1e-09).rsqrt()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
