import torch
import torch.nn as nn


class ResUnit(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.norm_1 = nn.InstanceNorm2d(in_channels)
        self.norm_2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.ELU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            dilation=dilation, padding=dilation)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            dilation=dilation, padding=dilation)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.norm_1(x)
        out = self.activation(out)
        out = self.conv_1(out)
        out = self.norm_2(out)
        out = self.activation(out)
        out = self.conv_2(out)
        out = self.shortcut(x) + out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
