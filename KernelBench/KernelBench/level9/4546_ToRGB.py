import torch
import torch.nn as nn


class ToRGB(nn.Module):
    """Some Information about ToRGB"""

    def __init__(self, input_channels):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1,
            padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4}]
