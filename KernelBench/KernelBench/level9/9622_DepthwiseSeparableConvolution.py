import torch
from torch import nn


class DepthwiseSeparableConvolution(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        """
            input : N*C1
            output : N*C1
            groups = C1
        """
        self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=
            in_ch, kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_ch)
        """
            input : N*C1
            output : N*C2
            kernel_size = 1
            groups = 1
        """
        self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=
            out_ch, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
