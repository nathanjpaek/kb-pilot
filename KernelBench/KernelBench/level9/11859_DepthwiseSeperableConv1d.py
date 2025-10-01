import torch
import torch.nn as nn


class DepthwiseSeperableConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeperableConv1d, self).__init__()
        self.depthwise_conv1d = nn.Conv1d(in_channels, in_channels,
            kernel_size, groups=in_channels, padding=kernel_size // 2)
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv1d(x)
        x = self.pointwise_conv1d(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
