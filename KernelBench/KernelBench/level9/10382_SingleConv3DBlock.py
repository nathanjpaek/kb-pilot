import torch
from torch import nn
import torch._utils


class SingleConv3DBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=
            kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.block(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}]
