import torch
import torch.nn as nn
from torch.quantization import QuantStub
from torch.quantization import DeQuantStub


class UpsamplingBilinear(nn.Module):

    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        upsample = nn.functional.interpolate(x, scale_factor=2, mode=
            'bilinear', align_corners=True)
        return self.dequant(upsample)

    def fuse_model(self):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
