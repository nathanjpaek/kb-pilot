import torch
import torch.nn as nn


class CSDN_Tem(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
        dilation=1):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
            kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
