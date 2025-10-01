import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import InstanceNorm2d


class mySConv(nn.Module):

    def __init__(self, num_filter=128, stride=1, in_channels=128):
        super(mySConv, self).__init__()
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, stride=
            stride, padding=1, in_channels=in_channels)
        self.bn = InstanceNorm2d(num_features=num_filter)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class mySBlock(nn.Module):

    def __init__(self, num_filter=128):
        super(mySBlock, self).__init__()
        self.myconv = mySConv(num_filter=num_filter, stride=1, in_channels=
            num_filter)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, padding=
            1, in_channels=num_filter)
        self.bn = InstanceNorm2d(num_features=num_filter)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(x + self.bn(self.conv(self.myconv(x))))


def get_inputs():
    return [torch.rand([4, 128, 64, 64])]


def get_init_inputs():
    return [[], {}]
