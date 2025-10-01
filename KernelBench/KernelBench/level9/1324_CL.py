import torch
import torch.nn as nn


class CL(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(CL, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride
            =2, padding=2)
        self.lr = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        out = self.cnn(x)
        out = self.lr(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
