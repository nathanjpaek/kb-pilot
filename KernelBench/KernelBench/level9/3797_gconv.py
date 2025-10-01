import torch
import torch.nn as nn
import torch.utils.model_zoo


class gconv(nn.Module):

    def __init__(self, channel):
        super(gconv, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        y = y + x
        y = self.relu(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
