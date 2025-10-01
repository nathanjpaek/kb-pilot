import torch
from torch import nn


class ConvDownsample(nn.Module):
    """Convolutional Downsampling of ConvMLP."""

    def __init__(self, embed_dim_in, embed_dim_out):
        super().__init__()
        self.downsample = nn.Conv2d(embed_dim_in, embed_dim_out, 3, stride=
            2, padding=1)

    def forward(self, x):
        """Forward function."""
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        return x.permute(0, 2, 3, 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim_in': 4, 'embed_dim_out': 4}]
