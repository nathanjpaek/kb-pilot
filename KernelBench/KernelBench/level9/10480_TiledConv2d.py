import torch
import torch.nn as nn
import torch.nn.functional as F


class TiledConv2d(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3,
            bias=False)

    def forward(self, x):
        return self.conv(F.pad(x, [1, 1, 1, 1], mode='circular'))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
