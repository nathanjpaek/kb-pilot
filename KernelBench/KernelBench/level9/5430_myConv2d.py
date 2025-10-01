import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class myConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return F.conv2d(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, _bias = ctx.saved_tensors
        _out_channels, _in_channels, kernel_height, kernel_width = list(weight
            .size())
        grad_input = F.conv2d(grad_output, torch.Tensor.rot90(weight, 2, [2,
            3]).transpose(0, 1), padding=(kernel_width - 1, kernel_height - 1))
        grad_weight = F.conv2d(input.transpose(0, 1), grad_output.transpose
            (0, 1)).transpose(0, 1)
        grad_bias = grad_output.sum([0, 2, 3])
        return grad_input, grad_weight, grad_bias


class myConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(myConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kenerl_size = kernel_size
        sqrtk = math.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
            kernel_size[0], kernel_size[1]))
        self.weight.data.uniform_(-sqrtk, sqrtk)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.bias.data.uniform_(-sqrtk, sqrtk)

    def forward(self, input):
        return myConv2dFunction.apply(input, self.weight, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': [4, 4]}]
