import torch
import numpy as np
import torch.nn as nn


class Sharpen_Block(nn.Module):

    def __init__(self):
        super(Sharpen_Block, self).__init__()
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.conv.weight = nn.Parameter(torch.from_numpy(np.array([[[[0, -
            0.4, 0], [0, 2.6, 0], [0, -0.4, 0]]]])).float())
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(self.pad(x))


def get_inputs():
    return [torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {}]
