import torch
import torch.nn as nn
from torch.nn import functional as F


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True,
    zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1,
    scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights,
        groups=groups, scaled=scaled)


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1,
    scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights,
        groups=groups, scaled=scaled)


class Block(nn.Module):

    def __init__(self, in_width, middle_width, out_width, down_rate=None,
        residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(
            middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(
            middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self
                .down_rate)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_width': 4, 'middle_width': 4, 'out_width': 4}]
