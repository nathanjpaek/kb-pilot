import torch
from torch import nn


class Predict(nn.Module):

    def __init__(self, in_planes=32, out_planes=1, kernel_size=1):
        super(Predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size)

    def forward(self, x):
        y = self.conv(x)
        return y


def get_inputs():
    return [torch.rand([4, 32, 64, 64])]


def get_init_inputs():
    return [[], {}]
