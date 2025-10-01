import torch
import torch.nn as M


class Upsample(M.Module):

    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = M.Upsample(scale_factor=2, mode='bilinear',
            align_corners=True)
        self.ordinaryConv = M.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.ordinaryConv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
