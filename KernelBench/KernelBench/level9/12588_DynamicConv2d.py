import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn


class DynamicConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, sr_in_list=(1.0,),
        sr_out_list=None):
        self.sr_idx, self.sr_in_list = 0, sorted(set(sr_in_list), reverse=True)
        if sr_out_list is not None:
            self.sr_out_list = sorted(set(sr_out_list), reverse=True)
        else:
            self.sr_out_list = self.sr_in_list
        super(DynamicConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups=groups, bias=bias)

    def forward(self, input):
        in_channels = round(self.in_channels * self.sr_in_list[self.sr_idx])
        out_channels = round(self.out_channels * self.sr_out_list[self.sr_idx])
        weight, bias = self.weight[:out_channels, :in_channels, :, :], None
        if self.bias is not None:
            bias = self.bias[:out_channels]
        return F.conv2d(input, weight, bias, self.stride, self.padding,
            self.dilation, round(self.groups * self.sr_in_list[self.sr_idx]
            ) if self.groups > 1 else 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
