import torch
import torch.nn as nn
import torch.utils.data


class GatedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, groups=1):
        super(GatedConv, self).__init__()
        self.layer_f = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=1, groups=groups)
        self.layer_g = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=1, groups=groups)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
