import torch
import torch.nn as nn


class unet_bottleneck(nn.Module):

    def __init__(self, in_ch, out_ch, ker=3):
        super(unet_bottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.bn1 = nn.GroupNorm(out_ch // 4, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
        self.bn2 = nn.GroupNorm(out_ch // 4, out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 1)
        self.bn3 = nn.GroupNorm(out_ch // 4, out_ch)
        self.s_conv = nn.Conv2d(in_ch, out_ch, 3)
        self.s_bn = nn.GroupNorm(out_ch // 4, out_ch)

    def forward(self, x):
        xp = self.conv1(x)
        xp = self.bn1(xp)
        xp = self.relu(xp)
        xp = self.conv2(xp)
        xp = self.bn2(xp)
        xp = self.relu(xp)
        xp = self.conv3(xp)
        xp = self.bn3(xp)
        x = self.s_conv(x)
        x = self.s_bn(x)
        return self.relu(xp + x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
