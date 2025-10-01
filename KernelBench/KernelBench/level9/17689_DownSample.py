import torch
import torch.nn as M


def DepthwiseConv(in_channels, kernel_size, stride, padding):
    return M.Conv2d(in_channels=in_channels, out_channels=in_channels,
        kernel_size=kernel_size, stride=stride, padding=padding, groups=
        in_channels, bias=False)


def PointwiseConv(in_channels, out_channels):
    return M.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=1, padding=0, bias=True)


class CovSepBlock(M.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
        padding=2):
        super().__init__()
        self.dc = DepthwiseConv(in_channels, kernel_size, stride=stride,
            padding=padding)
        self.pc = PointwiseConv(in_channels, out_channels)

    def forward(self, x):
        x = self.dc(x)
        x = self.pc(x)
        return x


class DownSample(M.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ordinaryConv1 = CovSepBlock(in_channels=in_channels,
            out_channels=out_channels // 4, stride=2)
        self.activate = M.ReLU(inplace=True)
        self.ordinaryConv2 = CovSepBlock(in_channels=out_channels // 4,
            out_channels=out_channels)
        self.skipconnect = CovSepBlock(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.activate2 = M.ReLU(inplace=True)

    def forward(self, x):
        branch = x
        x = self.ordinaryConv1(x)
        x = self.activate(x)
        x = self.ordinaryConv2(x)
        branch = self.skipconnect(branch)
        x += branch
        x = self.activate2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
