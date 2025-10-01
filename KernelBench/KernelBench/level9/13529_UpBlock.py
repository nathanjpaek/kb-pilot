import torch
import torch.cuda
import torch.nn as nn


class UpBlock(nn.Module):

    def __init__(self, in_, out, scale):
        super().__init__()
        self.up_conv = nn.Conv2d(in_, out, 1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale)

    def forward(self, x):
        return self.upsample(self.up_conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_': 4, 'out': 4, 'scale': 1.0}]
