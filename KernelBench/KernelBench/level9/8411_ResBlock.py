import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_c):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding
            =1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding
            =1, bias=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_c': 4}]
