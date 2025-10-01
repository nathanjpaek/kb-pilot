from torch.autograd import Function
import torch
import torch.nn as nn


class Round(Function):

    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ActivationQuantizer(nn.Module):

    def __init__(self, a_bits):
        super(ActivationQuantizer, self).__init__()
        self.a_bits = a_bits

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            None
            assert self.a_bits != 1
        else:
            output = torch.clamp(input * 0.1, 0, 1)
            scale = 1 / float(2 ** self.a_bits - 1)
            output = self.round(output / scale) * scale
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'a_bits': 4}]
