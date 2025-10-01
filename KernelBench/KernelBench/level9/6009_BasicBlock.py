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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'outplanes': 4}]
