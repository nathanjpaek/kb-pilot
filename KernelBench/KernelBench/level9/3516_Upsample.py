import torch
from torch import nn


class Upsample(nn.Module):
    """
        Since the number of channels of the feature map changes after upsampling in HRNet.
        we have to write a new Upsample class.
    """

    def __init__(self, in_channels, out_channels, scale_factor, mode):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.instance = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.upsample(out)
        out = self.instance(out)
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'scale_factor': 1.0,
        'mode': 4}]
