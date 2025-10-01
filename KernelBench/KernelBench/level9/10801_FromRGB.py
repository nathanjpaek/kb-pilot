import torch
import torch.nn as nn


class FromRGB(nn.Module):
    """Some Information about FromRGB"""

    def __init__(self, channels):
        super(FromRGB, self).__init__()
        self.conv = nn.Conv2d(3, channels, kernel_size=1, stride=1, padding
            =0, bias=True)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'channels': 4}]
