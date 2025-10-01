import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetUpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNetUpsamplingBlock, self).__init__()
        params = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True}
        self.conv = nn.Conv2d(in_channels, out_channels, **params)
        self.relu = nn.ReLU(inplace=True)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        x = self.relu(x)
        x = self.instance_norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
