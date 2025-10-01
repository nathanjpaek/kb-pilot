import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConvTranspose3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        total_pad = tuple([(k - s) for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.convt = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, bias=bias, padding=tuple([(k - 1) for
            k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
