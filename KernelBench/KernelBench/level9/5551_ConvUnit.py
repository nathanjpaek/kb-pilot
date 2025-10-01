import torch
import torch.nn as nn


class ConvUnit(nn.Module):

    def __init__(self):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=256, out_channels=32, kernel_size
            =5, stride=1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
