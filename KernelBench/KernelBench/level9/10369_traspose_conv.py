import torch
import torch.nn as nn


class traspose_conv(nn.Module):

    def __init__(self, num_of_channels):
        super(traspose_conv, self).__init__()
        self.trasnpose_conv = nn.ConvTranspose2d(num_of_channels, int(
            num_of_channels / 2), kernel_size=2, stride=2)

    def forward(self, x):
        x = self.trasnpose_conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_of_channels': 4}]
