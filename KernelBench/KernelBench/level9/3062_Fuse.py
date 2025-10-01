import torch
import torch.nn as nn


class Fuse(nn.Module):

    def __init__(self):
        super(Fuse, self).__init__()
        self.convolution = nn.Conv2d(32, 16, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convolution(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 32, 64, 64])]


def get_init_inputs():
    return [[], {}]
