import torch
import torch.nn as nn


class DilatedBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, dilation=1):
        super(DilatedBasicBlock, self).__init__()
        padding_size = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        assert padding_size % 2 == 0
        padding_size = int(padding_size / 2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            stride=1, padding=padding_size, dilation=dilation)
        self.in1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
            stride=1, padding=padding_size, dilation=dilation)
        self.in2 = nn.InstanceNorm2d(planes)
        if inplanes != planes:
            self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1)
            self.in3 = nn.InstanceNorm2d(planes)
        else:
            self.conv3 = None
            self.in3 = None

    def forward(self, x):
        if self.conv3 is not None:
            skip = self.in3(self.conv3(x))
        else:
            skip = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += skip
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
