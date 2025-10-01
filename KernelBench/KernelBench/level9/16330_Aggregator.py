import torch
import torchvision.transforms.functional as F
from torch.nn import functional as F
from torch import nn


class Aggregator(nn.Module):

    def __init__(self, in_channels, mid_channels, upsample_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2 ** upsample_factor)
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.conv(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'mid_channels': 4, 'upsample_factor': 4}]
