import torch
from torchvision.transforms import *
import torch.nn as nn


class SRCNN(nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=
            9, padding=9 // 2)
        self.conv = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=
            5, padding=5 // 2)
        self.output = nn.Conv2d(in_channels=32, out_channels=3, kernel_size
            =5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.input.weight, gain=nn.init.
            calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.
            calculate_gain('relu'))
        nn.init.xavier_uniform_(self.output.weight, gain=nn.init.
            calculate_gain('relu'))

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.relu(self.conv(out))
        out = self.output(out)
        return out


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
