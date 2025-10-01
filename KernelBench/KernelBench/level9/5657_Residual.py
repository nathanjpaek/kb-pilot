import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    """Unlinke other blocks, this module receives unpadded inputs."""

    def __init__(self, channels, kernel_size=3):
        super(Residual, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.pad = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size)
        self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        h = self.pad(x)
        h = self.conv1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.pad(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = h + x
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
