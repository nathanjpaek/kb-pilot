import torch
import torch.nn as nn


class INDeConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, out_padding=0, dilation=1, groups=1, relu=True, ins_n=
        True, bias=False):
        super(INDeConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, output_padding=
            out_padding, dilation=dilation, groups=groups, bias=bias)
        self.ins_n = nn.InstanceNorm2d(out_planes, affine=True
            ) if ins_n else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.ins_n is not None:
            x = self.ins_n(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}]
