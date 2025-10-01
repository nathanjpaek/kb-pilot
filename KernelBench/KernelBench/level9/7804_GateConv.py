import torch
import torch.nn as nn


class GateConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, transpose=False):
        super(GateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels *
                2, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.gate_conv(x)
        x, g = torch.split(x, self.out_channels, dim=1)
        return x * torch.sigmoid(g)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
