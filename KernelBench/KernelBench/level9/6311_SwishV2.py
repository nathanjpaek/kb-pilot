import torch
import torch.nn as nn


class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feat):
        sig = torch.sigmoid(feat)
        out = feat * torch.sigmoid(feat)
        grad = sig * (1 + feat * (1 - sig))
        ctx.grad = grad
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.grad
        grad *= grad_output
        return grad


class SwishV2(nn.Module):

    def __init__(self):
        super(SwishV2, self).__init__()

    def forward(self, feat):
        return SwishFunction.apply(feat)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
