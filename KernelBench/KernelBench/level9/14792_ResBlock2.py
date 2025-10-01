import torch
import torch.nn as nn
import torch.multiprocessing


class ResBlock2(nn.Module):

    def __init__(self, input_feature, planes, dilated=1, group=1):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(input_feature, planes, kernel_size=1, bias=
            False, groups=group)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1 *
            dilated, bias=False, dilation=dilated, groups=group)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, input_feature, kernel_size=1, bias=
            False, groups=group)
        self.bn3 = nn.InstanceNorm2d(input_feature)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_feature': 4, 'planes': 4}]
