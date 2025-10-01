import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, droprate=0.2, attention=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,
            bias=False)
        self.in1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
            bias=False)
        self.in2 = nn.InstanceNorm2d(planes)
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
        if self.attention is not None:
            out = self.attention(out)
        out += skip
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
