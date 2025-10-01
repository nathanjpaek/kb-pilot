import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
from torch.autograd.function import once_differentiable
from torch.nn.parameter import Parameter
import torch.nn.parallel
import torch.optim
import torch.utils.data


class WeightQuantization(A.Function):

    @staticmethod
    def forward(ctx, weight: 'Tensor', alpha: 'Tensor') ->Tensor:
        ctx.save_for_backward(weight, alpha)
        return alpha * weight.sign()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: 'Tensor') ->Tensor:
        weight, alpha = ctx.saved_tensors
        grad_input = alpha * grad_output
        grad_alpha = grad_output * weight.sign()
        grad_alpha.unsqueeze_(0)
        return grad_input, grad_alpha


class QLinear(nn.Linear):

    def __init__(self, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.alpha = Parameter(torch.ones(1))

    def forward(self, input: 'Tensor') ->Tensor:
        quantized_weight = WeightQuantization.apply(self.weight, self.alpha)
        return F.linear(input, quantized_weight, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
