import torch
import torch.nn as nn


class SpRes(nn.Module):

    def __init__(self, in_channels=31):
        super(SpRes, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=31, out_channels=3, bias=False,
            kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.Tanh()(x)
        return x


def get_inputs():
    return [torch.rand([4, 31, 64, 64])]


def get_init_inputs():
    return [[], {}]
