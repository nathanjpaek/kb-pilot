import torch
import torch.nn as nn
import torch.cuda
import torch.backends.cudnn
import torch.backends.mkl


class ConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose2d, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels,
            out_channels, kernel_size, stride, padding, output_padding,
            groups, bias, dilation)

    def forward(self, x):
        x = self.conv_transpose2d(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
