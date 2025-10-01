import torch
import torch.nn as nn


class LayerNorm1D(nn.Module):

    def __init__(self, num_outputs, eps=1e-05, affine=True):
        super(LayerNorm1D, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_outputs))
        self.bias = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, inputs):
        input_mean = inputs.mean(1, keepdim=True).expand_as(inputs)
        input_std = inputs.std(1, keepdim=True).expand_as(inputs)
        x = (inputs - input_mean) / (input_std + self.eps)
        return x * self.weight.expand_as(x) + self.bias.expand_as(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_outputs': 4}]
