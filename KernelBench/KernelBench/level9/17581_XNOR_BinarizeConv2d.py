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


class XNOR_BinarizeConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
        binary_func='deter'):
        super(XNOR_BinarizeConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.binary_func = binary_func
        w = self.weight
        sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1
            ).detach()
        self.alpha = nn.Parameter(sw, requires_grad=True)

    def forward(self, input):
        a0 = input
        w = self.weight
        w1 = w - w.mean([1, 2, 3], keepdim=True)
        w2 = w1 / w1.std([1, 2, 3], keepdim=True)
        a1 = a0 - a0.mean([1, 2, 3], keepdim=True)
        a2 = a1 / a1.std([1, 2, 3], keepdim=True)
        bw = XNOR_BinaryQuantize().apply(w2)
        ba = XNOR_BinaryQuantize_a().apply(a2)
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        output = output * self.alpha
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
