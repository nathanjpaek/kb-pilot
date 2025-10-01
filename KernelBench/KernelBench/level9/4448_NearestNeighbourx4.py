import torch
import torch.nn as nn
import torch.nn.functional as F


class NearestNeighbourx4(nn.Module):

    def __init__(self, nf, bias, custom_init=False):
        super(NearestNeighbourx4, self).__init__()
        self.conv0 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        if custom_init:
            for conv in [self.conv0, self.conv1, self.conv2]:
                torch.nn.init.kaiming_normal_(conv.weight)

    def forward(self, x):
        x = self.relu(self.conv0(F.interpolate(x, scale_factor=2, mode=
            'nearest')))
        x = self.relu(self.conv1(F.interpolate(x, scale_factor=2, mode=
            'nearest')))
        x = self.relu(self.conv2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nf': 4, 'bias': 4}]
