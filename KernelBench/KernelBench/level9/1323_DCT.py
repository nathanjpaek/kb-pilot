import torch
import torch.nn as nn


class DCT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, out_pad=0):
        super(DCT, self).__init__()
        self.dcnn = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size=2, stride=2, output_padding=out_pad)
        self.tanh = nn.Tanh()

    def forward(self, x1):
        out = self.dcnn(x1)
        out = self.tanh(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
