import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    BasicBlock implementation for ResNet
    
    reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """
    expansion = 1

    def __init__(self, device, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.device = device
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=
            stride, padding=2, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def _forward(self, n, i, weights, x):
        out = F.conv2d(x, weights['layer{}.{}.conv1.weight'.format(n, i)],
            stride=self.stride, padding=2)
        out = F.relu(out)
        out = F.conv2d(out, weights['layer{}.{}.conv2.weight'.format(n, i)])
        conv = 0
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            conv = F.conv2d(x, weights['layer{}.{}.shortcut.0.weight'.
                format(n, i)], stride=self.stride)
            x += conv
        out = F.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0, 'in_planes': 4, 'planes': 4}]
