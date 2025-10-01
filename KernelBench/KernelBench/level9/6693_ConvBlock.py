import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd,
        padding=padding, bias=bias)


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, use_instance_norm):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes
            ) if use_instance_norm else nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes // 2))
        self.bn2 = nn.InstanceNorm2d(int(out_planes / 2)
            ) if use_instance_norm else nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.InstanceNorm2d(int(out_planes / 4)
            ) if use_instance_norm else nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.InstanceNorm2d(in_planes) if
                use_instance_norm else nn.BatchNorm2d(in_planes), nn.ReLU(
                True), nn.Conv2d(in_planes, out_planes, kernel_size=1,
                stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)
        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        return out3


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4, 'use_instance_norm': 4}]
