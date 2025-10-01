import torch
import torch.nn as nn
from torch.quantization import QuantStub
from torch.quantization import DeQuantStub


class Hsigmoid(nn.Module):

    def __init__(self, add_stub=False):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        if self.add_stub:
            x = self.quant(x)
        x = self.hsigmoid(x)
        if self.add_stub:
            x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
