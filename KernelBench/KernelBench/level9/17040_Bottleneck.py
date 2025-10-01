import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, droprate=0.2, attention=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1,
            bias=False)
        self.in1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
            bias=False)
        self.in3 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.skip_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            1, bias=False)
        self.skip_in = nn.InstanceNorm2d(planes)
        self.attention = attention
        self.droprate = droprate

    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.in1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.in3(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        if self.attention is not None:
            out = self.attention(out)
        skip = self.skip_conv(skip)
        skip = self.skip_in(skip)
        if self.droprate > 0:
            skip = F.dropout(skip, p=self.droprate, training=self.training)
        out += skip
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
