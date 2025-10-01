import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, filter_size, dilation, residual_channels,
        dilated_channels, skip_channels):
        super().__init__()
        self.conv = nn.Conv1d(residual_channels, dilated_channels,
            kernel_size=filter_size, padding=dilation * (filter_size - 1),
            dilation=dilation)
        self.res = nn.Conv1d(dilated_channels // 2, residual_channels, 1)
        self.skip = nn.Conv1d(dilated_channels // 2, skip_channels, 1)
        self.filter_size = filter_size
        self.dilation = dilation
        self.residual_channels = residual_channels

    def forward(self, x, condition):
        length = x.shape[2]
        h = self.conv(x)
        h = h[:, :, :length]
        h += condition
        tanh_z, sig_z = torch.split(h, h.size(1) // 2, dim=1)
        z = torch.tanh(tanh_z) * torch.sigmoid(sig_z)
        if x.shape[2] == z.shape[2]:
            residual = self.res(z) + x
        else:
            residual = self.res(z) + x[:, :, -1:]
        skip_connection = self.skip(z)
        return residual, skip_connection

    def initialize(self, n):
        self.queue = torch.zeros((n, self.residual_channels, self.dilation *
            (self.filter_size - 1) + 1), dtype=self.conv.weight.dtype)
        self.conv.padding = 0

    def pop(self, condition):
        return self(self.queue, condition)

    def push(self, x):
        self.queue = torch.cat((self.queue[:, :, 1:], x), dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'filter_size': 4, 'dilation': 1, 'residual_channels': 4,
        'dilated_channels': 4, 'skip_channels': 4}]
