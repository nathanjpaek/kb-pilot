import torch
import torch.nn as nn
from itertools import product as product


class TCB(nn.Module):
    """
    Transfer Connection Block Architecture
    This block
    """

    def __init__(self, lateral_channels, channles, internal_channels=256,
        is_batchnorm=False):
        """
        :param lateral_channels: number of forward feature channles
        :param channles: number of pyramid feature channles
        :param internal_channels: number of internal channels
        """
        super(TCB, self).__init__()
        self.is_batchnorm = is_batchnorm
        use_bias = not self.is_batchnorm
        self.conv1 = nn.Conv2d(lateral_channels, internal_channels,
            kernel_size=3, padding=1, bias=use_bias)
        self.conv2 = nn.Conv2d(internal_channels, internal_channels,
            kernel_size=3, padding=1, bias=use_bias)
        self.deconv = nn.ConvTranspose2d(channles, internal_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias
            )
        self.conv3 = nn.Conv2d(internal_channels, internal_channels,
            kernel_size=3, padding=1, bias=use_bias)
        self.relu = nn.ReLU(inplace=True)
        if self.is_batchnorm:
            self.bn1 = nn.BatchNorm2d(internal_channels)
            self.bn2 = nn.BatchNorm2d(internal_channels)
            self.deconv_bn = nn.BatchNorm2d(internal_channels)
            self.bn3 = nn.BatchNorm2d(internal_channels)
        self.out_channels = internal_channels

    def forward(self, lateral, x):
        if self.is_batchnorm:
            lateral_out = self.relu(self.bn1(self.conv1(lateral)))
            out = self.relu(self.bn2(self.conv2(lateral_out)) + self.
                deconv_bn(self.deconv(x)))
            out = self.relu(self.bn3(self.conv3(out)))
        else:
            lateral_out = self.relu(self.conv1(lateral))
            out = self.relu(self.conv2(lateral_out) + self.deconv(x))
            out = self.relu(self.conv3(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 8, 8])]


def get_init_inputs():
    return [[], {'lateral_channels': 4, 'channles': 4}]
