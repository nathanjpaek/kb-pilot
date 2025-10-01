import torch
from torch import nn


class EncoderSlot(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)

    def forward(self, inputs):
        out = self.conv_1(inputs)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
