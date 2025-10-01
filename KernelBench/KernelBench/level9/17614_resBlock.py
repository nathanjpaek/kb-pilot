import torch
import torch.nn as nn
import torch.nn.functional as F


class resBlock(nn.Module):

    def __init__(self, channelDepth, windowSize=3):
        super(resBlock, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.IN_conv1 = nn.InstanceNorm2d(channelDepth)
        self.conv1 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, 0)
        self.conv2 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, 0)

    def forward(self, x):
        res = x
        x = F.relu(self.IN_conv1(self.conv1(self.pad(x))))
        x = self.IN_conv1(self.conv2(self.pad(x)))
        x = x + res
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channelDepth': 4}]
