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


class GridReduction2(nn.Module):

    def __init__(self, in_channels):
        super(GridReduction2, self).__init__()
        self.branch3x3_1 = Conv2d(in_channels, 192, 1)
        self.branch3x3_2 = Conv2d(192, 320, 3, stride=2)
        self.branch7x7x3_1 = Conv2d(in_channels, 192, 1)
        self.branch7x7x3_2 = Conv2d(192, 192, (1, 7), padding=(0, 3))
        self.branch7x7x3_3 = Conv2d(192, 192, (7, 1), padding=(3, 0))
        self.branch7x7x3_4 = Conv2d(192, 192, 3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branches = [branch3x3, branch7x7x3, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
