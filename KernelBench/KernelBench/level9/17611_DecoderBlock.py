import torch
from functools import partial
import torch.nn.functional as F
from torch import nn


class DecoderBlock(nn.Module):
    """
    Decoder block class
    """

    def __init__(self, in_channels, middle_channels, out_channels, k_size,
        pad_size):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=
            k_size, padding=pad_size)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=
            k_size, padding=pad_size)
        self.IN1 = nn.InstanceNorm3d(out_channels)
        self.IN2 = nn.InstanceNorm3d(out_channels)
        self.upsample = partial(F.interpolate, scale_factor=2, mode=
            'trilinear', align_corners=False)

    def forward(self, x):
        x = F.leaky_relu(self.IN1(self.conv1(x)), inplace=True)
        x = F.leaky_relu(self.IN2(self.conv2(x)), inplace=True)
        x = self.upsample(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4,
        'k_size': 4, 'pad_size': 4}]
