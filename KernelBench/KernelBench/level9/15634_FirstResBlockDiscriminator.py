import torch
import numpy as np
from torch import nn
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm


class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0
            )
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x
        self.model = nn.Sequential(self.spec_norm(self.conv1), nn.ReLU(),
            self.spec_norm(self.conv2), nn.AvgPool2d(2))
        self.bypass = nn.Sequential(nn.AvgPool2d(2), self.spec_norm(self.
            bypass_conv))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
