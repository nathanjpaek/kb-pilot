import torch
import torch.nn as nn


class SampaddingConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, use_bias=True):
        super(SampaddingConv1D, self).__init__()
        self.use_bias = use_bias
        self.padding = nn.ConstantPad1d((int((kernel_size - 1) / 2), int(
            kernel_size / 2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels
            =out_channels, kernel_size=kernel_size, bias=self.use_bias)

    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        return X


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
