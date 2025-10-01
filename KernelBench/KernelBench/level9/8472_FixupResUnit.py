import torch
import torch.nn.functional as F
import torch.nn as nn


class FixupResUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1,
            stride=stride, bias=False)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
            bias=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=
                stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out + self.bias1a)
        out = out + self.bias1b
        out = F.relu(out)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b
        return self.shortcut(x) + out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
