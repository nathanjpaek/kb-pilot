import torch
import torch.nn as nn


def conv(in_channel, out_channel, kernel_size, stride=1, dilation=1, bias=False
    ):
    padding = (kernel_size - 1) * dilation // 2
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=bias)


class convBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1,
        dilation=1, bias=False, nonlinear=True, bn=False):
        super().__init__()
        self.conv = conv(in_channel, out_channel, kernel_size, stride,
            dilation, bias)
        self.nonlinear = nn.ReLU(inplace=True) if nonlinear else None
        self.bn = nn.BatchNorm2d(out_channel, eps=0.0001, momentum=0.95
            ) if bn else None

    def forward(self, x):
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.nonlinear is not None:
            out = self.nonlinear(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
