import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.quantized.modules import FloatFunctional


class TorchAddScalar(nn.Module):
    """ Wrapper around torch.add so that all ops can be found at build
        y must be a scalar, needed for quantization
    """

    def __init__(self):
        super().__init__()
        self.add_func = FloatFunctional()

    def forward(self, x, y):
        return self.add_func.add_scalar(x, y)


class TorchMulScalar(nn.Module):
    """Wrapper around torch.mul so that all ops can be found at build
        y must be a scalar, needed for quantization
    """

    def __init__(self):
        super().__init__()
        self.mul_func = FloatFunctional()

    def forward(self, x, y):
        return self.mul_func.mul_scalar(x, y)


class HSigmoid(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)
        self.add_scalar = TorchAddScalar()
        self.mul_scalar = TorchMulScalar()

    def forward(self, x):
        return self.mul_scalar(self.relu(self.add_scalar(x, 3.0)), 1.0 / 6.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
