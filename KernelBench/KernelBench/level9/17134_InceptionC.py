import torch
from torch.nn import functional as F
import torch.nn as nn


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=
        False, **kwargs):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionC(nn.Module):

    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.branch1x1 = Conv2d(in_channels, 320, 1)
        self.branch3x3_1 = Conv2d(in_channels, 384, 1)
        self.branch3x3_2a = Conv2d(384, 384, (1, 3), padding=(0, 1))
        self.branch3x3_2b = Conv2d(384, 384, (3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = Conv2d(in_channels, 448, 1)
        self.branch3x3dbl_2 = Conv2d(448, 384, 3, padding=1)
        self.branch3x3dbl_3a = Conv2d(384, 384, (1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = Conv2d(384, 384, (3, 1), padding=(1, 0))
        self.branch_pool = Conv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)
            ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.
            branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branches = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
