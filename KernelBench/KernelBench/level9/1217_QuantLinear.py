from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class QuantLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, a_bits=8,
        w_bits=8, quant_inference=False):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = ActivationQuantizer(a_bits=a_bits)
        self.weight_quantizer = WeightQuantizer(w_bits=w_bits)

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
