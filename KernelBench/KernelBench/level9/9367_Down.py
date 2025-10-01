import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, factor=2):
        super(Down, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=factor, padding=1)

    def forward(self, x):
        c = F.elu(self.down(x))
        return c


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
