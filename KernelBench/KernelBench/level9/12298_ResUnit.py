import torch
import torch.nn as nn


class ResUnit(nn.Module):

    def __init__(self, ksize=3, wkdim=64):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize / 2))
        self.active = nn.PReLU()
        self.conv2 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize / 2))

    def forward(self, input):
        current = self.conv1(input)
        current = self.active(current)
        current = self.conv2(current)
        current = input + current
        return current


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
