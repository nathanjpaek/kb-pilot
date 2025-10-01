import torch
from torch import nn


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-08)


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels,
        kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels * num_conv_units, kernel_size=kernel_size, stride=
            stride)
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.
            out_channels), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_conv_units': 4, 'in_channels': 4, 'out_channels': 4,
        'kernel_size': 4, 'stride': 1}]
