import torch
import torch.nn as M


class Upsampling(M.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.upsample = M.ConvTranspose2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=2)

    def forward(self, x):
        return self.upsample(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
