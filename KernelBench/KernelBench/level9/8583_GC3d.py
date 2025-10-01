import torch
import torch.nn as nn
import torch.nn


class GC3d(nn.Module):

    def __init__(self, inplanes, planes, kh=7, kw=7, mdim=256, which_conv=
        nn.Conv3d):
        super(GC3d, self).__init__()
        self.conv_l1 = which_conv(inplanes, mdim, kernel_size=(1, kh, 1),
            padding=(0, int(kh / 2), 0))
        self.conv_l2 = which_conv(mdim, planes, kernel_size=(1, 1, kw),
            padding=(0, 0, int(kw / 2)))
        self.conv_r1 = which_conv(inplanes, mdim, kernel_size=(1, 1, kw),
            padding=(0, 0, int(kw / 2)))
        self.conv_r2 = which_conv(mdim, planes, kernel_size=(1, kh, 1),
            padding=(0, int(kh / 2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
