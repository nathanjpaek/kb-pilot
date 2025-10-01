import torch
import torch.nn as nn


class QuantizeAct(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, numbits):
        ctx.save_for_backward(input)
        if numbits == 1:
            return input.sign()
        elif numbits == 2:
            return torch.floor(input + 0.5)
        else:
            return torch.floor(input.add(1).div(2).clamp_(0, 0.999).mul(2 **
                numbits - 1)).sub((2 ** numbits - 1) // 2)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class Quantizer(nn.Module):

    def __init__(self, numbits):
        super(Quantizer, self).__init__()
        self.numbits = numbits

    def forward(self, input):
        return QuantizeAct.apply(input, self.numbits)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'numbits': 4}]
