import torch
import torch.utils.data
import torch.nn as nn


class DiracInitBlock(nn.Module):
    """
    DiracNetV2 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(DiracInitBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=7, stride=2, padding=3, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
