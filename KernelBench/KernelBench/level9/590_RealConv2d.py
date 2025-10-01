import torch
import torch.nn as nn
import torch.nn.functional as F


class RealConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1),
        stride=(1, 1), padding=(0, 0), dilation=1, groups=1, causal=True,
        complex_axis=1):
        """
            in_channels: real+imag
            out_channels: real+imag
            kernel_size : input [B,C,D,T] kernel size in [D,T]
            padding : input [B,C,D,T] padding in [D,T]
            causal: if causal, will padding time dimension's left side,
                    otherwise both

        """
        super(RealConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
            kernel_size, self.stride, padding=[self.padding[0], 0],
            dilation=self.dilation, groups=self.groups)
        nn.init.normal_(self.conv.weight.data, std=0.05)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])
        out = self.conv(inputs)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
