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


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = Conv2d(in_channels, 64, 1)
        self.branch5x5_1 = Conv2d(in_channels, 48, 1)
        self.branch5x5_2 = Conv2d(48, 64, 5, padding=2)
        self.branch3x3dbl_1 = Conv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = Conv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = Conv2d(96, 96, 3, padding=1)
        self.branch_pool = Conv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branches = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'pool_features': 4}]
