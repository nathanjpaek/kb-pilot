from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F


class XNOR_BinaryQuantize(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().clamp(min=-1, max=1)
        return grad_input


class XNOR_BinaryQuantize_a(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.sign(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[1].le(-1)] = 0
        return grad_input


class binary_last_fc(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(binary_last_fc, self).__init__(in_features, out_features, bias)
        w = self.weight
        sw = w.abs().mean().float().detach()
        self.alpha = nn.Parameter(sw, requires_grad=True)

    def forward(self, input):
        a0 = input
        w = self.weight
        w1 = w - w.mean([1], keepdim=True)
        w2 = w1 / w1.std([1], keepdim=True)
        a1 = a0 - a0.mean([1], keepdim=True)
        a2 = a1 / a1.std([1], keepdim=True)
        bw = XNOR_BinaryQuantize().apply(w2)
        ba = XNOR_BinaryQuantize_a().apply(a2)
        output = F.linear(ba, bw, self.bias)
        output = output * self.alpha
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
