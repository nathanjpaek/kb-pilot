from torch.nn import Module
import torch
from typing import Any
import torch.nn.functional as F


class TauSTEFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Any', tau_threshold: 'float', input: 'Any') ->Any:
        return (input > tau_threshold).float()

    @staticmethod
    def backward(ctx: 'Any', grad_output: 'Any') ->Any:
        return None, F.hardtanh(grad_output)


class TauSTE(Module):

    def __init__(self, tau_threshold: 'float'=0.0) ->None:
        super(TauSTE, self).__init__()
        self.tau_threshold = tau_threshold

    def forward(self, batch: 'torch.Tensor') ->torch.Tensor:
        return TauSTEFunction.apply(self.tau_threshold, batch)

    def extra_repr(self) ->str:
        return 'tau_threshold={}'.format(self.tau_threshold)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
