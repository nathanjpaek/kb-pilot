import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.quantized.modules import FloatFunctional


class TorchMulScalar(nn.Module):
    """Wrapper around torch.mul so that all ops can be found at build
        y must be a scalar, needed for quantization
    """

    def __init__(self):
        super().__init__()
        self.mul_func = FloatFunctional()

    def forward(self, x, y):
        return self.mul_func.mul_scalar(x, y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
