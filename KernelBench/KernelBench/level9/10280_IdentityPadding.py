import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityPadding(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()
        self.pooling = nn.MaxPool2d(1, stride=stride)
        self.add_channels = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'stride': 1}]
