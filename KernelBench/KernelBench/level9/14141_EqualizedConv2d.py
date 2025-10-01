import math
import torch
import torch.nn as nn
import torch.utils.cpp_extension


@torch.no_grad()
def scaling_init(tensor, scale=1, dist='u'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    scale /= (fan_in + fan_out) / 2
    if dist == 'n':
        std = math.sqrt(scale)
        return tensor.normal_(0.0, std)
    elif dist == 'u':
        bound = math.sqrt(3 * scale)
        return tensor.uniform_(-bound, bound)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=
    True, scale=1.0):
    _conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias)
    scaling_init(_conv.weight, scale)
    if _conv.bias is not None:
        nn.init.zeros_(_conv.bias)
    return _conv


class EqualizedLR(nn.Module):
    """
    equalized learning rate
    """

    def __init__(self, layer, gain=2):
        super(EqualizedLR, self).__init__()
        self.wscale = (gain / layer.weight[0].numel()) ** 0.5
        self.layer = layer

    def forward(self, x, gain=2):
        x = self.layer(x * self.wscale)
        return x


class EqualizedConv2d(nn.Module):
    """
    equalized convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        conv.weight.data.normal_(0, 1)
        conv.bias.data.fill_(0.0)
        self.conv = EqualizedLR(conv)

    def forward(self, x):
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
