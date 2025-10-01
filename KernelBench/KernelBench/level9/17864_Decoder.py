import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = [((i - 1) // 2) for i in kernel_size]
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.tconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': [4, 4],
        'stride': 1}]
