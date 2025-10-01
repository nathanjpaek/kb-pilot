import torch
import torch.nn as nn
import torch.quantization


class QuantizableHSigmoid(nn.Module):
    """Hard Sigmoid for quantization."""

    def __init__(self, inplace: 'bool'=True) ->None:
        """Initialize."""
        super(QuantizableHSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
        self.add_scalar = nn.quantized.FloatFunctional()
        self.mul_scalar = nn.quantized.FloatFunctional()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward."""
        x = self.add_scalar.add_scalar(x, 3.0)
        x = self.relu6(x)
        x = self.mul_scalar.mul_scalar(x, 1 / 6)
        return x


class QuantizableHSwish(nn.Module):
    """Hard Swish for quantization."""

    def __init__(self, inplace: 'bool'=True) ->None:
        """Initialize."""
        super(QuantizableHSwish, self).__init__()
        self.hsig = QuantizableHSigmoid(inplace=inplace)
        self.mul = nn.quantized.FloatFunctional()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward."""
        return self.mul.mul(x, self.hsig(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
