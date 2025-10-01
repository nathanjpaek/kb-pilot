import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed


class AdaptiveConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=1,
        dilation=1, groups=1, bias=False, size=(256, 256)):
        super(AdaptiveConv, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride,
            padding=1, dilation=dilation, groups=groups, bias=bias)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride,
            padding=0, dilation=dilation, groups=groups, bias=bias)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.size = size
        self.w = nn.Parameter(torch.ones(3, 1, self.size[0], self.size[1]))
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, _h, _w = x.size()
        weight = self.softmax(self.w)
        w1 = weight[0, :, :, :]
        w2 = weight[1, :, :, :]
        w3 = weight[2, :, :, :]
        x1 = self.conv3x3(x)
        x2 = self.conv1x1(x)
        size = x1.size()[2:]
        gap = self.gap(x)
        gap = self.relu(self.fc1(gap))
        gap = self.fc2(gap)
        gap = F.upsample(gap, size=size, mode='nearest')
        x = w1 * x1 + w2 * x2 + w3 * gap
        return x


def get_inputs():
    return [torch.rand([4, 4, 256, 256])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
