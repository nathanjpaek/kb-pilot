import torch
import torch.nn as nn


class BasicConv3D(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv3D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.InstanceNorm3d(out_planes, eps=1e-05, momentum=0.01,
            affine=True)
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool3D(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1)
            .unsqueeze(1)), dim=1)


class SpatialGate3D(nn.Module):

    def __init__(self):
        super(SpatialGate3D, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool3D()
        self.spatial = BasicConv3D(2, 1, kernel_size, stride=1, padding=(
            kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


def get_inputs():
    return [torch.rand([4, 2, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
