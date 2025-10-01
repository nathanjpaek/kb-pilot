import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """Upsample block for DRRG and TextSnake."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1,
            stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels,
            kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1x1(x))
        x = F.relu(self.conv3x3(x))
        x = self.deconv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
