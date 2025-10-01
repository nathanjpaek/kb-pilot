import torch
from torch import nn


class DecoderBlock(nn.Module):
    """Decoder block"""

    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2,
        padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size,
            stride, padding)
        self.bn = nn.InstanceNorm2d(outplanes)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)

    def forward(self, x):
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)
        if self.dropout is not None:
            fx = self.dropout(fx)
        return fx


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'outplanes': 4}]
