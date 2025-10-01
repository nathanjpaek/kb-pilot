import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=True)


class Resv1Block(nn.Module):
    """ResNet v1 block without bn"""

    def __init__(self, inplanes, planes, stride=1):
        super(Resv1Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += x
        out = self.relu2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
