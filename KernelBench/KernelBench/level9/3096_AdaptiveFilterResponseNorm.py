import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.jit
import torch.nn


class AdaptiveFilterResponseNorm(nn.Module):

    def __init__(self, in_size, ada_size, eps=1e-16):
        super().__init__()
        self.eps = eps
        self.in_size = in_size
        self.scale = nn.Linear(ada_size, in_size)
        self.bias = nn.Linear(ada_size, in_size)
        self.threshold = nn.Linear(ada_size, in_size)

    def forward(self, inputs, condition):
        out = inputs.view(inputs.size(0), inputs.size(1), -1)
        nu2 = out.mean(dim=-1)
        extension = [1] * (inputs.dim() - 2)
        denominator = torch.sqrt(nu2 + self.eps)
        denominator = denominator.view(inputs.size(0), inputs.size(1), *
            extension)
        out = inputs / denominator
        scale = self.scale(condition)
        bias = self.bias(condition)
        threshold = self.threshold(condition)
        out = func.relu(scale * out + bias - threshold) + threshold
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'ada_size': 4}]
