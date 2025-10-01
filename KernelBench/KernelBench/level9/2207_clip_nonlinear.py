import torch
import torch.nn as nn


def quantize_a(x):
    x = Q_A.apply(x)
    return x


def fa(x, bitA):
    if bitA == 32:
        return x
    return quantize_a(x)


class Q_A(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = x.new(x.size())
        out[x > 0] = 1
        out[x < 0] = -1
        out[x == 0] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input > 1.0, 0.0)
        grad_input.masked_fill_(input < -1.0, 0.0)
        mask_pos = (input >= 0.0) & (input < 1.0)
        mask_neg = (input < 0.0) & (input >= -1.0)
        grad_input.masked_scatter_(mask_pos, input[mask_pos].mul_(-2.0).
            add_(2.0))
        grad_input.masked_scatter_(mask_neg, input[mask_neg].mul_(2.0).add_
            (2.0))
        return grad_input * grad_output


class clip_nonlinear(nn.Module):

    def __init__(self, bitA):
        super(clip_nonlinear, self).__init__()
        self.bitA = bitA

    def forward(self, input):
        return fa(input, self.bitA)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'bitA': 4}]
