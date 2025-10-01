import torch
import torch.nn as nn


class _SubPixelBlock(nn.Module):

    def __init__(self, in_channels: 'int'=64, out_channels: 'int'=64,
        scale_factor: 'int'=2):
        super(_SubPixelBlock, self).__init__()
        n_out = out_channels * scale_factor ** 2
        self.conv = nn.Conv2d(in_channels, n_out, kernel_size=3, stride=1,
            padding=1)
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        hid = self.conv(x)
        hid = self.shuffle(hid)
        out = self.prelu(hid)
        return out


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
