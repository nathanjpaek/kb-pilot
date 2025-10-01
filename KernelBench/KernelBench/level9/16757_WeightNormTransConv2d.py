import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.onnx


class WeightNormTransConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, output_padding=0, dilation=1, groups=1, bias=True,
        padding_mode='zeros'):
        super(WeightNormTransConv2d, self).__init__()
        self.conv = weight_norm(nn.ConvTranspose2d(in_channels,
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode, output_padding=output_padding))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(
                input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        x = self.conv(input)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
