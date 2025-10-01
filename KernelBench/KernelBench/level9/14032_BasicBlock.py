import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=0, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, weightnorm=None,
        shortcut=True):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu1 = nn.PReLU(num_parameters=planes, init=0.1)
        self.relu2 = nn.PReLU(num_parameters=planes, init=0.1)
        self.conv2 = conv3x3(inplanes, planes, stride)
        if weightnorm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)

    def forward(self, x):
        out = self.relu1(x)
        out = F.pad(out, (1, 1, 1, 1), 'reflect')
        out = self.conv1(out)
        out = out[:, :, :x.shape[2], :x.shape[3]]
        out = self.relu2(out)
        out = F.pad(out, (1, 1, 1, 1), 'reflect')
        out = self.conv2(out)
        out = out[:, :, :x.shape[2], :x.shape[3]]
        if self.shortcut:
            out = x + out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
