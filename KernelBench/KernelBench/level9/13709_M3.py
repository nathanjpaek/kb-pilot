import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, same_padding
        =False, stride=1, relu=True, bn=False):
        super(Conv2D, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0,
            affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class M3(nn.Module):

    def __init__(self, in_channels):
        super(M3, self).__init__()
        self.m3_ssh_3x3 = Conv2D(in_channels, 256, 3, True, 1, False)
        self.m3_ssh_dimred = Conv2D(in_channels, 128, 3, True, 1, True)
        self.m3_ssh_5x5 = Conv2D(128, 128, 3, True, 1, False)
        self.m3_ssh_7x7_1 = Conv2D(128, 128, 3, True, 1, True)
        self.m3_ssh_7x7 = Conv2D(128, 128, 3, True, 1, False)
        self.m3_ssh_cls_score = Conv2D(128 * 2 + 256, 4, 1, False, 1, False)
        self.m3_ssh_bbox_pred = Conv2D(128 * 2 + 256, 8, 1, False, 1, False)

    def forward(self, pool6):
        m3_ssh_3x3 = self.m3_ssh_3x3(pool6)
        m3_ssh_dimred = self.m3_ssh_dimred(pool6)
        m3_ssh_5x5 = self.m3_ssh_5x5(m3_ssh_dimred)
        m3_ssh_7x7_1 = self.m3_ssh_7x7_1(m3_ssh_dimred)
        m3_ssh_7x7 = self.m3_ssh_7x7(m3_ssh_7x7_1)
        m3_ssh_output = F.relu(torch.cat((m3_ssh_3x3, m3_ssh_5x5,
            m3_ssh_7x7), dim=1))
        m3_ssh_cls_score = self.m3_ssh_cls_score(m3_ssh_output)
        m3_ssh_bbox_pred = self.m3_ssh_bbox_pred(m3_ssh_output)
        return m3_ssh_cls_score, m3_ssh_bbox_pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
