import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        padding=0, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
        padding=2, bias=False)


def conv9x9(in_planes, out_planes, stride=1):
    """9x9 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=9, stride=stride,
        padding=4, bias=False)


def maxpool3x3(stride=2):
    """3x3 maxpooling with padding"""
    return nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)


class FirstStage(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, out_channels, stride=1):
        super(FirstStage, self).__init__()
        self.out_channels = out_channels
        self.conv1 = conv9x9(3, planes)
        self.pool1 = maxpool3x3()
        self.conv2 = conv9x9(planes, planes)
        self.pool2 = maxpool3x3()
        self.conv3 = conv9x9(planes, planes)
        self.pool3 = maxpool3x3()
        self.conv4 = conv5x5(planes, planes // self.expansion)
        self.conv5 = conv9x9(planes // self.expansion, planes * self.expansion)
        self.conv6 = conv1x1(planes * self.expansion, planes * self.expansion)
        self.conv7 = conv1x1(planes * self.expansion, self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)
        out = self.conv7(out)
        return out


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4, 'out_channels': 4}]
