import torch
import torch.nn as nn
import torch.jit
import torch.nn


class NotNorm(nn.Module):

    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size

    def forward(self, inputs):
        [1] * (inputs.dim() - 2)
        out = inputs.view(inputs.size(0), inputs.size(1), -1)
        mean = out.mean(dim=-1, keepdim=True)
        std = out.std(dim=-1, keepdim=True)
        normed = (out - mean.detach()) / std.detach()
        out = std * normed + mean
        return out.view(inputs.shape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
