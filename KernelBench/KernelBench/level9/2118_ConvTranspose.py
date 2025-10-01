import torch
import torch.nn as nn


class ConvTranspose(nn.Module):
    """Convolution Module with transposes of last two dimensions."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=True, w_init='relu'):
        super(ConvTranspose, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.
            calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
