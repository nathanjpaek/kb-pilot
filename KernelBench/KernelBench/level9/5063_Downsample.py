import torch
import torch.nn as nn
import torch.hub


class Downsample(nn.Module):

    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels,
                kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'with_conv': 4}]
