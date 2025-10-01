import torch
import torch.nn as nn


def Maxout(x1, x2, x3, x4):
    mask_1 = torch.ge(x1, x2)
    mask_1 = mask_1.float()
    x = mask_1 * x1 + (1 - mask_1) * x2
    mask_2 = torch.ge(x, x3)
    mask_2 = mask_2.float()
    x = mask_2 * x + (1 - mask_2) * x3
    mask_3 = torch.ge(x, x4)
    mask_3 = mask_3.float()
    x = mask_3 * x + (1 - mask_3) * x4
    return x


class DC_layer(nn.Module):

    def __init__(self, level, fuse=False):
        super(DC_layer, self).__init__()
        self.level = level
        self.conv1x1_d1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d2 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d3 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d4 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv_d1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1
            )
        self.conv_d2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2
            )
        self.conv_d3 = nn.Conv2d(512, 512, kernel_size=3, padding=3, dilation=3
            )
        self.conv_d4 = nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4
            )
        self.fuse = fuse
        if self.fuse:
            self.fuse = nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1x1_d1(x)
        x2 = self.conv1x1_d2(x)
        x3 = self.conv1x1_d3(x)
        x4 = self.conv1x1_d4(x)
        x1 = self.conv_d1(x1)
        x2 = self.conv_d2(x2)
        x3 = self.conv_d3(x3)
        x4 = self.conv_d4(x4)
        x = Maxout(x1, x2, x3, x4)
        return x


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {'level': 4}]
