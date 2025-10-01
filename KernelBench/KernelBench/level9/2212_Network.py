import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
