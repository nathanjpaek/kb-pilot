import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock_AP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock_AP, self).__init__()
        self.norm = norm
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True
            ) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True
            ) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2), nn.GroupNorm(self.
                expansion * planes, self.expansion * planes, affine=True) if
                self.norm == 'instancenorm' else nn.BatchNorm2d(self.
                expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.stride != 1:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'planes': 4}]
