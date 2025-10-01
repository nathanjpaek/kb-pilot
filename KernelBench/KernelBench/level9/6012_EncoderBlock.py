import torch
from torch import nn


class EncoderBlock(nn.Module):
    """Encoder block"""

    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2,
        padding=1, norm=True, padding_mode='zeros'):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride,
            padding, padding_mode=padding_mode)
        self.bn = None
        if norm:
            self.bn = nn.InstanceNorm2d(outplanes)

    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)
        if self.bn is not None:
            fx = self.bn(fx)
        return fx


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'outplanes': 4}]
