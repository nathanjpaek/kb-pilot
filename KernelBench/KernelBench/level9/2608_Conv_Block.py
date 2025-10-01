import torch
from torchvision.transforms import *
import torch.nn as nn


class Conv_Block(nn.Module):

    def __init__(self):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=
            3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.
            calculate_gain('relu'))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
