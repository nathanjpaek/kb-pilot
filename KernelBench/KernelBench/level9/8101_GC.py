import torch
import torch.nn as nn
import torch.nn.parallel


class GC(nn.Module):

    def __init__(self, inplanes, planes, kh=7, kw=7):
        super(GC, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, 256, kernel_size=(kh, 1),
            padding=(int(kh / 2), 0))
        self.conv_l2 = nn.Conv2d(256, planes, kernel_size=(1, kw), padding=
            (0, int(kw / 2)))
        self.conv_r1 = nn.Conv2d(inplanes, 256, kernel_size=(1, kw),
            padding=(0, int(kw / 2)))
        self.conv_r2 = nn.Conv2d(256, planes, kernel_size=(kh, 1), padding=
            (int(kh / 2), 0))

    def forward(self, x):
        x_l = self.conv_l2(self.conv_l1(x))
        x_r = self.conv_r2(self.conv_r1(x))
        x = x_l + x_r
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
