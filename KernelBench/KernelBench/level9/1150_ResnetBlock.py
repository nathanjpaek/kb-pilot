import torch
import torch.nn as nn


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn=False):
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.bn4 = nn.BatchNorm2d(out_channels)
            self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(x)
        out = self.pool1(out)
        identity = out
        out = self.relu(out)
        if self.use_bn:
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out)) + identity
        else:
            out = self.relu(self.conv2(out))
            out = self.conv3(out) + identity
        identity = out
        out = self.relu(out)
        if self.use_bn:
            out = self.relu(self.bn4(self.conv4(out)))
            out = self.bn5(self.conv5(out)) + identity
        else:
            out = self.relu(self.conv4(out))
            out = self.conv5(out) + identity
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
