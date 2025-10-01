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


class ActQuant_PACT(nn.Module):

    def __init__(self, act_bit=4, scale_coef=1.0):
        super(ActQuant_PACT, self).__init__()
        self.act_bit = act_bit
        self.scale_coef = nn.Parameter(torch.ones(1) * scale_coef)
        self.uniform_q = uniform_quantize(k=act_bit)

    def forward(self, x):
        if self.act_bit == 32:
            out = 0.5 * (x.abs() - (x - self.scale_coef.abs()).abs() + self
                .scale_coef.abs()) / self.scale_coef.abs()
        else:
            out = 0.5 * (x.abs() - (x - self.scale_coef.abs()).abs() + self
                .scale_coef.abs())
            activation_q = self.uniform_q(out / self.scale_coef)
        return activation_q


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
