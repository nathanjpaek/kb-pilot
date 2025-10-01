import torch
import torch.nn as nn


class ComplexConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else
            'cpu')
        self.padding = padding
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias)

    def forward(self, x):
        real = self.conv_re(x[:, 0]) - self.conv_im(x[:, 1])
        imaginary = self.conv_re(x[:, 1]) + self.conv_im(x[:, 0])
        output = torch.stack((real, imaginary), dim=1)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}]
