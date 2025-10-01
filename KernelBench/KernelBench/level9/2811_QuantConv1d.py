import torch
from torch import nn


class QuantConv1d(nn.Module):
    """Quantized 1D Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
        **kwargs):
        super().__init__()
        self.qconv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias, padding_mode=padding_mode, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.qconv1d(x)
        x = self.dequant(x)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
