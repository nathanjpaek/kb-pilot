import torch
import torch.onnx
import torch
import torch.nn as nn
import torch.utils.data


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1):
        super(ConvBlock, self).__init__()
        self.Mconv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding)
        self.MPrelu = nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        x = self.Mconv(x)
        x = self.MPrelu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
