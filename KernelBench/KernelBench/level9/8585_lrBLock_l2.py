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


class lrBLock_l2(nn.Module):

    def __init__(self, channelDepth, windowSize=3):
        super(lrBLock_l2, self).__init__()
        math.floor(windowSize / 2)
        self.res_l2 = resBlock(channelDepth, windowSize)
        self.res_l1 = resBlock(channelDepth, windowSize)

    def forward(self, x):
        x_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_reup = F.interpolate(x_down2, scale_factor=2, mode='bilinear')
        Laplace_1 = x - x_reup
        Scale1 = self.res_l1(x_down2)
        Scale2 = self.res_l2(Laplace_1)
        output2 = F.interpolate(Scale1, scale_factor=2, mode='bilinear'
            ) + Scale2
        return output2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channelDepth': 4}]
