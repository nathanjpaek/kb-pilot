from torch.autograd import Function
import torch
from typing import Optional
from typing import Tuple
import torch.nn as nn
from typing import Any
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.optim


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: 'Any', input: 'torch.Tensor', coeff: 'Optional[float]'=1.0
        ) ->torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: 'Any', grad_output: 'torch.Tensor') ->Tuple[torch.
        Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):

    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class Theta(nn.Module):
    """
    maximize loss respect to :math:`	heta`
    minimize loss respect to features
    """

    def __init__(self, dim: 'int'):
        super(Theta, self).__init__()
        self.grl1 = GradientReverseLayer()
        self.grl2 = GradientReverseLayer()
        self.layer1 = nn.Linear(dim, dim)
        nn.init.eye_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)

    def forward(self, features: 'torch.Tensor') ->torch.Tensor:
        features = self.grl1(features)
        return self.grl2(self.layer1(features))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
