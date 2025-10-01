import torch
import torch.nn as nn
import torch.nn.functional as F


class RemoveChannelMeanStd(torch.nn.Module):

    def forward(self, x):
        x2 = x.view(x.size(0), x.size(1), -1)
        mean = x2.mean(dim=2).view(x.size(0), x.size(1), 1, 1)
        std = x2.std(dim=2).view(x.size(0), x.size(1), 1, 1)
        return (x - mean) / std


class Block(nn.Module):

    def __init__(self, in_planes, planes, stride=1, groups=False):
        super(Block, self).__init__()
        self.bn1 = RemoveChannelMeanStd()
        if not groups:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride
                =stride, padding=1, bias=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride
                =stride, padding=1, bias=True, groups=min(in_planes, planes))
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                padding=1, bias=True, groups=planes)
        self.bn2 = RemoveChannelMeanStd()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,
                kernel_size=1, stride=stride, bias=True),
                RemoveChannelMeanStd())

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        relu1 = F.relu(out)
        out = self.conv2(relu1)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out += shortcut
        out = F.relu(out)
        return relu1, out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'planes': 4}]
