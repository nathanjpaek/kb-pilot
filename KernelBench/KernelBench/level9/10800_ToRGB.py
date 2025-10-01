import torch
import torch.nn as nn


class ToRGB(nn.Module):
    """Some Information about ToRGB"""

    def __init__(self, channels):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding
            =0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
