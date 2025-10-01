import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Redisual network block for style transfer."""

    def __init__(self, nchannels):
        """Create a block of a residual network."""
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, nchannels, kernel_size=3)
        self.conv2 = nn.Conv2d(nchannels, nchannels, kernel_size=3)
        self.norm_conv1 = nn.InstanceNorm2d(nchannels, affine=True)
        self.norm_conv2 = nn.InstanceNorm2d(nchannels, affine=True)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        """Forward the input through the block."""
        residual = x[:, :, 2:-2, 2:-2]
        out = self.nonlinearity(self.norm_conv1(self.conv1(x)))
        out = self.norm_conv2(self.conv2(out))
        return out + residual


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'nchannels': 4}]
