import torch
import torch.nn as nn
import torch.utils.model_zoo


class BasicConv(nn.Module):

    def __init__(self, in_feature, out_feature, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = BatchNorm2d(out_feature, eps=1e-05, momentum=0.01, affine
            =True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'out_feature': 4, 'kernel_size': 4}]
