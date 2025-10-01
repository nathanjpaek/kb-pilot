import torch
import torch.nn as nn


class sSEmodule(nn.Module):
    """ ChannelSequeezeExcitationModule
        input: [B, C, H, W] torch tensor
        output: [B, C, H, W] torch tensor
    """

    def __init__(self, in_channel):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connection = x
        x = self.conv2d(x)
        x = self.sigmoid(x)
        None
        x = x * skip_connection
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4}]
