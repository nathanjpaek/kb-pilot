import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class resBlock(nn.Module):

    def __init__(self, channelDepth, windowSize=3):
        super(resBlock, self).__init__()
        padding = math.floor(windowSize / 2)
        self.conv1 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1,
            padding)
        self.conv2 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1,
            padding)
        self.IN_conv = nn.InstanceNorm2d(channelDepth, track_running_stats=
            False, affine=False)

    def forward(self, x):
        res = x
        x = F.relu(self.IN_conv(self.conv1(x)))
        x = self.IN_conv(self.conv2(x))
        x = F.relu(x + res)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channelDepth': 4}]
