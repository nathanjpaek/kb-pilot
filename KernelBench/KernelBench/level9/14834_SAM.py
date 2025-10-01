import torch
from torch import nn


class SAM(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        spatial_features = self.conv(x)
        attention = torch.sigmoid(spatial_features)
        return attention.expand_as(x) * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
