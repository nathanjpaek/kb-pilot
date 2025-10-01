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


class WeightQuantizer(nn.Module):

    def __init__(self, w_bits):
        super(WeightQuantizer, self).__init__()
        self.w_bits = w_bits

    def round(self, input):
        output = Round.apply(input)
        return output

    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            None
            assert self.w_bits != 1
        else:
            output = torch.tanh(input)
            output = output / 2 / torch.max(torch.abs(output)) + 0.5
            scale = 1 / float(2 ** self.w_bits - 1)
            output = self.round(output / scale) * scale
            output = 2 * output - 1
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'w_bits': 4}]
