import torch
import torch.nn as nn


class ConvStem2(nn.Module):

    def __init__(self, in_chans=3, out_chans=64, kernel_size=7, stride=2):
        super(ConvStem2, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
