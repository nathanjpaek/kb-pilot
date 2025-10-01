import torch
import torch.nn as nn


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]
    if isinstance(stride, int):
        stride = [stride]
    assert len(stride) == len(kernel_size
        ), 'Pass kernel size and stride both as int, or both as equal length iterable'
    return [(((k - 1) * s + 1) // 2) for k, s in zip(kernel_size, stride)]


class Conv2dZeros(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
        stride=(1, 1), padding='same', logscale_factor=3):
        super().__init__()
        if padding == 'same':
            padding = compute_same_pad(kernel_size, stride)
        elif padding == 'valid':
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
