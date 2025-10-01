import torch
from torch import nn


class ResidualBlockNoBN(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(3, 3), stride=stride, padding=1,
            bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=
            out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=(1, 1), stride=
                stride, bias=False))

    def forward(self, x):
        out = nn.ReLU()(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
