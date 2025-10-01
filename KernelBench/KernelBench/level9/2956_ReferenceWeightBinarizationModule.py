import torch
from torch import nn
from torchvision import models as models
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx


class ReferenceDOREFABinarize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        norm = x.abs().mean()
        sign = (x > 0).type(x.dtype) * 2 - 1
        output_flat = sign * norm
        return output_flat.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ReferenceXNORBinarize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        norm = x.abs().mean([1, 2, 3], keepdim=True)
        sign = (x > 0).type(x.dtype) * 2 - 1
        output = sign * norm
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ReferenceWeightBinarizationModule(nn.Module):

    def __init__(self, mode='xnor'):
        super().__init__()
        self.mode = mode
        if self.mode == 'xnor':
            self.binarize = ReferenceXNORBinarize.apply
        elif self.mode == 'dorefa':
            self.binarize = ReferenceDOREFABinarize.apply

    def forward(self, input_):
        return self.binarize(input_)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
