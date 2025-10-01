import torch
import torch.nn as nn


def uniform_quantize(k):


    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return qfn().apply


class weight_quantize_fn(nn.Module):

    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit - 1)

    def forward(self, x):
        if self.w_bit == 32:
            weight = torch.tanh(x)
            weight_q = weight / torch.max(torch.abs(weight))
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = (self.uniform_q(x / E) + 1) / 2 * E
        else:
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q = self.uniform_q(weight)
        return weight_q


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'w_bit': 4}]
