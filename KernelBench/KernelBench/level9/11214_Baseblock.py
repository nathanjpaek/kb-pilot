import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class Baseblock(nn.Module):

    def __init__(self, in_channels):
        super(Baseblock, self).__init__()
        self.p_size = [1, 1, 1, 1]
        self.pool1 = nn.MaxPool2d(kernel_size=self.p_size[0], stride=self.
            p_size[0])
        self.pool2 = nn.MaxPool2d(kernel_size=self.p_size[1], stride=self.
            p_size[1])
        self.pool3 = nn.MaxPool2d(kernel_size=self.p_size[2], stride=self.
            p_size[2])
        self.pool4 = nn.MaxPool2d(kernel_size=self.p_size[3], stride=self.
            p_size[3])
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1,
            dilation=2, kernel_size=3, padding=2)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x_size = x.size()
        layer0 = self.conv(x)
        layer1 = F.interpolate(self.conv(self.pool1(x)), size=x_size[2:],
            mode='bilinear', align_corners=True)
        layer2 = F.interpolate(self.conv(self.pool2(x)), size=x_size[2:],
            mode='bilinear', align_corners=True)
        layer3 = F.interpolate(self.conv(self.pool3(x)), size=x_size[2:],
            mode='bilinear', align_corners=True)
        layer4 = F.interpolate(self.conv(self.pool4(x)), size=x_size[2:],
            mode='bilinear', align_corners=True)
        return layer0, layer1, layer2, layer3, layer4


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
