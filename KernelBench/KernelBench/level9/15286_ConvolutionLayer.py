import torch
import torch.nn as nn


class ConvolutionLayer(nn.Module):

    def __init__(self, channels, filters, kernel_size, stride=1, dilation=1):
        super(ConvolutionLayer, self).__init__()
        padding = kernel_size // 2
        padding += padding * (dilation - 1)
        self.conv = nn.Conv1d(channels, filters, kernel_size, stride=stride,
            dilation=dilation, padding=padding)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'filters': 4, 'kernel_size': 4}]
