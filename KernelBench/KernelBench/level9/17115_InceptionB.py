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


class InceptionB(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionB, self).__init__()
        self.branch1x1 = Conv2d(in_channels, 192, 1)
        c7 = channels_7x7
        self.branch7x7_1 = Conv2d(in_channels, c7, 1)
        self.branch7x7_2 = Conv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7_3 = Conv2d(c7, 192, (7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = Conv2d(in_channels, c7, 1)
        self.branch7x7dbl_2 = Conv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = Conv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = Conv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = Conv2d(c7, 192, (1, 7), padding=(0, 3))
        self.branch_pool = Conv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branches = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'channels_7x7': 4}]
