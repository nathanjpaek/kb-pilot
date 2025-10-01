import torch
import torch.nn as nn
from torch.quantization import QuantStub
from torch.quantization import DeQuantStub


class MulScalarNegative(nn.Module):

    def __init__(self):
        super().__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        mul = self.float_op.mul_scalar(x, -0.3)
        return self.dequant(mul)

    def fuse_model(self):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
