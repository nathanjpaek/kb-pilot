import torch
import torch.nn as nn
import torch.utils.data


class Quantization(nn.Module):

    @staticmethod
    def forward(input):
        return torch.round(input)

    @staticmethod
    def backward(grad_output):
        grad_input = grad_output.clone()
        return grad_input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
