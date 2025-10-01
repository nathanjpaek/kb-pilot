import torch
from torch import nn


class QuantPointwiseConv(nn.Module):
    """Quantized Point-wise Conv"""

    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__()
        self.qlinear = nn.Linear(in_channels, out_channels, bias=bias)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = x.transpose(1, 2)
        x = self.qlinear(x)
        x = x.transpose(1, 2)
        x = self.dequant(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
