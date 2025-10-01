import torch
from torch import nn


class BasicBlock(nn.Module):
    """Basic block"""

    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2,
        padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding
            )
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.conv(x)
        if self.isn is not None:
            fx = self.isn(fx)
        fx = self.lrelu(fx)
        return fx


class Discriminator(nn.Module):
    """Basic Discriminator"""

    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(3, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        return fx


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
