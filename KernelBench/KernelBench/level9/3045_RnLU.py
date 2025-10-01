import math
import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction
import torch.nn.parallel
import torch.utils.data


def birelu(x, inplace=False):
    return BiReLUFunction().apply(x, inplace)


def rnlu(x, inplace=False, shift=0, scale_fix=(math.pi / 2) ** 0.5):
    x = birelu(x, inplace=inplace)
    pos, neg = (x + shift).chunk(2, dim=1)
    scale = (pos - neg).view(pos.size(0), -1).mean(1) * scale_fix + 1e-08
    return x / scale.view(scale.size(0), *([1] * (x.dim() - 1)))


class BiReLUFunction(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, inplace=False):
        if input.size(1) % 2 != 0:
            raise RuntimeError(
                'dimension 1 of input must be multiple of 2, but got {}'.
                format(input.size(1)))
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        pos, neg = output.chunk(2, dim=1)
        pos.clamp_(min=0)
        neg.clamp_(max=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_variables
        grad_input = grad_output.masked_fill(output.eq(0), 0)
        return grad_input, None


class RnLU(nn.Module):
    """docstring for RnLU."""

    def __init__(self, inplace=False):
        super(RnLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return rnlu(x, inplace=self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
