import torch
import torch.nn as nn


class Conv3x3(nn.Module):

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, inputs):
        inputs = self.pad(inputs)
        out = self.conv(inputs)
        return out


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.upsamp = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, inputs):
        out = self.upsamp(inputs)
        out = self.conv(out)
        out = self.elu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
