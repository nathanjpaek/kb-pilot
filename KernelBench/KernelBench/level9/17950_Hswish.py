import torch
import torch.nn as nn
from torch.quantization import QuantStub
from torch.quantization import DeQuantStub


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True, add_stub=False):
        super().__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.relu6 = nn.ReLU6(inplace=inplace)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub

    def forward(self, x):
        if self.add_stub:
            x = self.quant(x)
        relu6 = self.relu6(self.float_op.add_scalar(x, 3.0))
        mul = self.float_op.mul_scalar(relu6, 1 / 6.0)
        if self.add_stub:
            mul = self.dequant(mul)
        return mul

    def fuse_model(self):
        pass


class Hswish(nn.Module):

    def __init__(self, inplace=True, add_stub=False):
        super(Hswish, self).__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.hsigmoid = Hsigmoid(inplace, add_stub=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub

    def forward(self, x):
        if self.add_stub:
            x = self.quant(x)
        mul = self.float_op.mul(x, self.hsigmoid(x))
        if self.add_stub:
            mul = self.dequant(mul)
        return mul

    def fuse_model(self):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
