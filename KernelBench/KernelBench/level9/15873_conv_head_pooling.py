import torch
from torch import nn


class conv_head_pooling(nn.Module):

    def __init__(self, in_feature, out_feature, stride, padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()
        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride +
            1, padding=stride // 2, stride=stride, padding_mode=
            padding_mode, groups=in_feature)

    def forward(self, x):
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'out_feature': 4, 'stride': 1}]
