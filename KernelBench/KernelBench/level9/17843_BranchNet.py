import torch
import torch.nn as nn
from itertools import product as product
from math import sqrt as sqrt
import torch.utils.data


def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, 1, bias=True)


class BranchNet(nn.Module):
    """
    The branch of NaiveNet is the network output and
    only consists of conv 1×1 and ReLU.
    """

    def __init__(self, inplanes, planes):
        super(BranchNet, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2_score = conv1x1(planes, planes)
        self.conv3_score = conv1x1(planes, 2)
        self.conv2_bbox = conv1x1(planes, planes)
        self.conv3_bbox = conv1x1(planes, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out_score = self.conv2_score(out)
        out_score = self.relu(out_score)
        out_score = self.conv3_score(out_score)
        out_bbox = self.conv2_bbox(out)
        out_bbox = self.relu(out_bbox)
        out_bbox = self.conv3_bbox(out_bbox)
        return out_score, out_bbox


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
