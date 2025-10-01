import torch
import torch.nn as nn
from torch.quantization import QuantStub
from torch.quantization import DeQuantStub


class Hswish(nn.Module):

    def __init__(self, add_stub=False):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub
        self.hswish = nn.Hardswish()

    def forward(self, x):
        if self.add_stub:
            x = self.quant(x)
        x = self.hswish(x)
        if self.add_stub:
            x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
