import torch
import torch.nn as nn
import torch.nn.functional


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1_ = nn.Conv3d(in_channels, mid_channels, kernel_size=3,
            padding=1)
        self.relu1_ = nn.ReLU(inplace=True)
        self.conv2_ = nn.Conv3d(mid_channels, out_channels, kernel_size=3,
            padding=1)
        self.relu2_ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1_(x)
        x = self.relu1_(x)
        x = self.conv2_(x)
        x = self.relu2_(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
