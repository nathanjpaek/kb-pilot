from torch.nn import Module
import torch
from torch.nn import ConvTranspose2d


class ComplexConvTranspose2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, groups=1, bias=True, dilation=1,
        padding_mode='zeros'):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride, padding, output_padding, groups, bias,
            dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride, padding, output_padding, groups, bias,
            dilation, padding_mode)

    def forward(self, input_r, input_i):
        return self.conv_tran_r(input_r) - self.conv_tran_i(input_i
            ), self.conv_tran_r(input_i) + self.conv_tran_i(input_r)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
