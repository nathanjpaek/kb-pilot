import torch
from torch import nn
import torch.nn.functional as F


class OnnxToTorchModule:
    """
    Marker class for onnx2torch modules.
    """
    pass


class OnnxGeneralLinear(nn.Linear, OnnxToTorchModule):
    """General Linear layer with functionality of ONNX GEMM node.

    For additional info https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
    """

    def __init__(self, in_features: 'int', out_features: 'int', bias:
        'bool', trans_a: 'int'):
        super().__init__(in_features=in_features, out_features=out_features,
            bias=bias)
        self.trans_a = trans_a

    def forward(self, input_tensor: 'torch.Tensor') ->torch.Tensor:
        input_tensor = torch.transpose(input_tensor, 0, 1
            ) if self.trans_a != 0 else input_tensor
        return F.linear(input_tensor, self.weight, self.bias)

    @classmethod
    def maybe_create_simple_linear(cls, in_features: 'int', out_features:
        'int', bias: 'bool', trans_a: 'int'):
        if trans_a == 0:
            return nn.Linear(in_features=in_features, out_features=
                out_features, bias=bias)
        return OnnxGeneralLinear(in_features, out_features, bias, trans_a)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'bias': 4, 'trans_a': 4}]
