from torch.autograd import Function
import torch
from torch import nn


def swish(x, beta=1.0):
    """Swish activation.
    'https://arxiv.org/pdf/1710.05941.pdf'
    Args:
        x: Input tensor.
        beta:
    """
    return SwishOP.apply(x, beta)


class SwishOP(Function):

    @staticmethod
    def forward(ctx, tensor, beta=1.0):
        ctx.save_for_backward(tensor)
        ctx.beta = beta
        swish = tensor / (1 + torch.exp(-beta * tensor))
        return swish

    @staticmethod
    def backward(ctx, grad_outputs):
        tensor = ctx.saved_tensors[0]
        beta = ctx.beta
        grad_swish = (torch.exp(-beta * tensor) * (1 + beta * tensor) + 1) / (
            1 + torch.exp(-beta * tensor)) ** 2
        grad_swish = grad_outputs * grad_swish
        return grad_swish, None


class Swish(nn.Module):
    """Switch activation from 'SEARCHING FOR ACTIVATION FUNCTIONS'
        https://arxiv.org/pdf/1710.05941.pdf

        swish =  x / (1 + e^-beta*x)
        d_swish = (1 + (1+beta*x)) / ((1 + e^-beta*x)^2)

    """

    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return swish(x, self.beta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
