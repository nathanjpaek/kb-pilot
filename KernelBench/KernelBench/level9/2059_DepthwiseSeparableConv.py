import torch
import torch.cuda
from torch.nn import functional as F
from torch import nn


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
        activation=F.relu):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channels,
            out_channels=in_channels, kernel_size=kernel_size, padding=
            kernel_size // 2, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channels,
            out_channels=out_channels, padding=0, kernel_size=1, bias=bias)
        self.activation = activation

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.activation:
            x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
