import torch
from torch import nn


class DecoderSlot(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.ConvTranspose2d(in_channels=66, out_channels=64,
            kernel_size=5, stride=(2, 2))
        self.conv_2 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
            kernel_size=5, stride=(2, 2))
        self.conv_3 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
            kernel_size=5, stride=(2, 2))
        self.conv_4 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
            kernel_size=5, stride=(2, 2))
        self.conv_5 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
            kernel_size=2)
        self.conv_6 = nn.ConvTranspose2d(in_channels=64, out_channels=4,
            kernel_size=2)

    def forward(self, inputs):
        out = self.conv_1(inputs)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.conv_6(out)
        return out


def get_inputs():
    return [torch.rand([4, 66, 4, 4])]


def get_init_inputs():
    return [[], {}]
