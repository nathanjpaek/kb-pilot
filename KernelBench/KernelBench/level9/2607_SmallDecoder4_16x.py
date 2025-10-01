import torch
import torch.nn as nn


class SmallDecoder4_16x(nn.Module):

    def __init__(self):
        super(SmallDecoder4_16x, self).__init__()
        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

    def forward(self, y):
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.conv11(self.pad(y))
        return y


def get_inputs():
    return [torch.rand([4, 128, 4, 4])]


def get_init_inputs():
    return [[], {}]
