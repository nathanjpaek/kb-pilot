import sys
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class ClassWisePoolFunction(Function):

    @staticmethod
    def forward(ctx, input, num_maps):
        batch_size, num_channels, h, w = input.size()
        if num_channels % num_maps != 0:
            None
            sys.exit(-1)
        num_outputs = int(num_channels / num_maps)
        x = input.view(batch_size, num_outputs, num_maps, h, w)
        output = torch.sum(x, 2)
        ctx.save_for_backward(input)
        ctx.num_maps = num_maps
        return output.view(batch_size, num_outputs, h, w) / num_maps

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        batch_size, num_channels, h, w = input.size()
        num_outputs = grad_output.size(1)
        grad_input = grad_output.view(batch_size, num_outputs, 1, h, w).expand(
            batch_size, num_outputs, ctx.num_maps, h, w).contiguous()
        return grad_input.view(batch_size, num_channels, h, w), None


class ClassWisePool(nn.Module):

    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps

    def forward(self, input):
        return ClassWisePoolFunction.apply(input, self.num_maps)

    def __repr__(self):
        return self.__class__.__name__ + ' (num_maps={num_maps})'.format(
            num_maps=self.num_maps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_maps': 4}]
