import torch
import torch.nn as nn


class IdentityMappingZero(nn.Module):

    def __init__(self, out_channels: 'int', stride: 'int') ->None:
        super(IdentityMappingZero, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        pad_value = self.out_channels // 4
        self.zeropad = nn.ZeroPad2d(padding=(0, 0, 0, 0, pad_value, pad_value))

    def forward(self, x):
        x = x[:, :, ::self.stride, ::self.stride]
        x = self.zeropad(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_channels': 4, 'stride': 1}]
