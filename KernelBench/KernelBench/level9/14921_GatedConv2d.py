import torch
import torch.nn as nn


class GatedConv2d(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride, pad, dilation=1, act=
        torch.relu):
        super(GatedConv2d, self).__init__()
        self.activation = act
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation)
        self.g = nn.Conv2d(in_c, out_c, kernel, stride, pad, dilation)

    def forward(self, x):
        h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_c': 4, 'out_c': 4, 'kernel': 4, 'stride': 1, 'pad': 4}]
