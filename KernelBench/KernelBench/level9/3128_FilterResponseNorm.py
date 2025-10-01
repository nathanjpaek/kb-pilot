import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.jit
import torch.nn


class FilterResponseNorm(nn.Module):

    def __init__(self, in_size, eps=1e-16):
        super().__init__()
        self.eps = eps
        self.in_size = in_size
        self.register_parameter('scale', nn.Parameter(torch.ones(in_size,
            dtype=torch.float)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(in_size,
            dtype=torch.float)))
        self.register_parameter('threshold', nn.Parameter(torch.zeros(
            in_size, dtype=torch.float)))

    def forward(self, inputs):
        out = inputs.view(inputs.size(0), inputs.size(1), -1)
        nu2 = (out ** 2).mean(dim=-1)
        extension = [1] * (inputs.dim() - 2)
        denominator = torch.sqrt(nu2 + self.eps)
        denominator = denominator.view(inputs.size(0), inputs.size(1), *
            extension)
        scale = self.scale.view(1, self.scale.size(0), *extension)
        bias = self.bias.view(1, self.bias.size(0), *extension)
        threshold = self.threshold.view(1, self.threshold.size(0), *extension)
        out = inputs / denominator.detach()
        out = func.relu(scale * out + bias - threshold) + threshold
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
