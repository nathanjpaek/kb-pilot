import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.prelu2 = nn.PReLU(planes)

    def forward(self, x):
        return x + self.prelu2(self.conv2(self.prelu1(self.conv1(x))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'planes': 4}]
