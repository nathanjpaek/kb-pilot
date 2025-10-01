import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """ Encoder  - From pyramid bottom to op
    """

    def __init__(self, in_channels, out_channels, sz=1):
        super(UpBlock, self).__init__()
        self.c1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1)
        self.c2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
            stride=(sz, 2, 2), padding=1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = F.leaky_relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
