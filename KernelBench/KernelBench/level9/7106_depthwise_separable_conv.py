import torch
import torch.nn as nn


class depthwise_separable_conv(torch.nn.Module):

    def __init__(self, nin, nout, kernel_size, padding):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size,
            padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nin': 4, 'nout': 4, 'kernel_size': 4, 'padding': 4}]
