import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPoolStride1(nn.Module):

    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x_pad = F.pad(x, (0, 1, 0, 1), mode='replicate')
        x = F.max_pool2d(x_pad, 2, stride=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
