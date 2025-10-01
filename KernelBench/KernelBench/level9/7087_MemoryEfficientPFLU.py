from torch.autograd import Function
import torch
from torch import nn


class PFLUFunction(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * (1 + x / torch.sqrt(1 + x * x)) / 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            t = 1 / (1 + x * x)
            grad_x = grad_output * (1 + x * torch.sqrt(t) * (1 + t)) / 2
        return grad_x


class MemoryEfficientPFLU(nn.Module):

    def forward(self, x):
        return PFLUFunction.apply(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
