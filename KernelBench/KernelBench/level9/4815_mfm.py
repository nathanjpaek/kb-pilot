import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class mfm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, mode=1):
        """
        mfm
        :param in_channels: in channel
        :param out_channels: out channel
        :param kernel_size: conv kernel size
        :param stride: conv stride
        :param padding: conv padding
        :param mode: 1: Conv2d  2: Linear
        """
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if mode == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
