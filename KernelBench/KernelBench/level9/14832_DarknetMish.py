import torch
import torch.nn.functional as F
from torch import nn


class darknet_mish(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        e = torch.exp(input)
        n = e * e + 2 * e
        mask = input <= -0.6
        input[mask] = (input * (n / (n + 2)))[mask]
        input[~mask] = (input - 2 * (input / (n + 2)))[~mask]
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        sp = F.softplus(input)
        grad_sp = -torch.expm1(sp)
        tsp = F.tanh(sp)
        grad_tsp = (1 - tsp * tsp) * grad_sp
        grad = input * grad_tsp + tsp
        return grad


class DarknetMish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return darknet_mish.apply(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
