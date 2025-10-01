from torch.autograd import Function
import torch
import torch.nn as nn


class SignFunction(Function):

    def __init__(self):
        super(SignFunction, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training=True):
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Sign(nn.Module):

    def __init__(self):
        super(Sign, self).__init__()

    def forward(self, x):
        return SignFunction.apply(x, self.training)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
