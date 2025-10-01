import torch
from torch import Tensor
import torch.nn as nn


class UpSampleAndHalveChannels(nn.Module):
    """
    Doubles the spatial dimensions (H,W) but halves the number of channels.
    Inverse of the DownSample function in blocks.py
    
    From Diakogiannis et al.
    doi: 10.1016/j.isprsjprs.2020.01.013
    """

    def __init__(self, _in_channels, _factor=2):
        super().__init__()
        self.in_channels = _in_channels
        self.factor = _factor
        self.upSample = nn.Upsample(scale_factor=self.factor, mode=
            'bilinear', align_corners=None)
        self.halveChannels = nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.in_channels // self.factor, kernel_size=(1, 1
            ), stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x: 'Tensor') ->Tensor:
        out = self.upSample(x)
        out = self.halveChannels(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'_in_channels': 4}]
