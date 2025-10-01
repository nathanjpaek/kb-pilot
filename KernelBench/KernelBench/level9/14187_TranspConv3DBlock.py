import torch
import torch.nn as nn


class TranspConv3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=
            2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4}]
