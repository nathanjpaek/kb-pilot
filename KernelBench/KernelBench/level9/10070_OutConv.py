import torch
import torch.nn as nn


class OutConv(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, input_):
        return self.tanh(self.conv(input_))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inChannels': 4, 'outChannels': 4}]
