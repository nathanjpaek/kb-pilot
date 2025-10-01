import torch
from torch import nn


class CubeFunctionBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, M):
        ctx.save_for_backward(X, M)
        return M * 3 * X ** 2

    @staticmethod
    def backward(ctx, V):
        X, M = ctx.saved_tensors
        return V * 6 * X * M, V * 3 * X ** 2


class CubeFunction(torch.autograd.Function):
    """
    Dummy activation function x -> x ** 3
    """

    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        return X ** 3

    @staticmethod
    def backward(ctx, M):
        X, = ctx.saved_tensors
        return CubeFunctionBackward.apply(X, M)


class Cube(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return CubeFunction.apply(X)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
