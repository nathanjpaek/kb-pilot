import torch
import torch.nn as nn


class Mfm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, f_type=1):
        super(Mfm, self).__init__()
        self.out_channels = out_channels
        if f_type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class Group(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding
        ):
        super(Group, self).__init__()
        self.conv_a = Mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = Mfm(in_channels, out_channels, kernel_size, stride, padding
            )

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
