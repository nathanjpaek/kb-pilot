import torch
import torch.nn as nn
import torch.nn.parallel


class SepConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels,
            kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity(
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += (H * W * self.in_channels * self.kernel_size ** 2 / self.
            stride ** 2)
        flops += H * W * self.in_channels * self.out_channels
        return flops


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
