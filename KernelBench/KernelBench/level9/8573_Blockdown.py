import torch
import torch.utils.data
import torch
import torch.nn as nn


class conv_bn_relu(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, has_relu=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=stride,
            padding=1, bias=True)
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = self.relu(x)
        return x


class Blockdown(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Blockdown, self).__init__()
        self.conv1 = conv_bn_relu(in_channel, out_channel, stride=2)
        self.conv2 = conv_bn_relu(out_channel, out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
