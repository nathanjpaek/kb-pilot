import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data


class TransferConv3(nn.Module):

    def __init__(self, n_channels, n_channels_in=None, residual=False):
        super().__init__()
        if n_channels_in is None:
            n_channels_in = n_channels
        self.conv1 = nn.Conv2d(n_channels_in, n_channels, kernel_size=3,
            stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3,
            stride=1, padding=1)
        self.residual = residual
        self.n_channels = n_channels

    def forward(self, x):
        x_copy = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.residual:
            if x.shape != x_copy.shape:
                x_copy = x_copy[:, :self.n_channels, :, :]
            x = x + x_copy
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
