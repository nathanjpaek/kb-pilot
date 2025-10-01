import torch
import torch.nn as nn
import torch.jit
import torch.nn


class AdaptiveLayerNorm(nn.Module):

    def __init__(self, in_size, ada_size):
        super(AdaptiveLayerNorm, self).__init__()
        self.scale = nn.Linear(ada_size, in_size)
        self.bias = nn.Linear(ada_size, in_size)

    def forward(self, inputs, style):
        expand = [1] * (inputs.dim() - 2)
        mean = inputs.mean(dim=1, keepdim=True)
        std = inputs.std(dim=1, keepdim=True)
        scale = self.scale(style).view(style.size(0), -1, *expand)
        scale = scale - scale.mean(dim=1, keepdim=True) + 1
        bias = self.bias(style).view(style.size(0), -1, *expand)
        bias = bias - bias.mean(dim=1, keepdim=True)
        return scale * (inputs - mean) / (std + 1e-06) + bias


def get_inputs():
    return [torch.rand([4, 64, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'ada_size': 4}]
