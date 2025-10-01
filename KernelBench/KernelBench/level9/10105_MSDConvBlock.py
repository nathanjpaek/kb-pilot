import torch
import torch.nn as nn


class MSDConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, std):
        super(MSDConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(3, 3), padding=(dilation, dilation),
            padding_mode='reflect', dilation=dilation, bias=False)
        torch.nn.init.normal_(self.conv.weight, 0, std)

    def forward(self, x):
        y = self.conv(x)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'dilation': 1, 'std': 4}]
