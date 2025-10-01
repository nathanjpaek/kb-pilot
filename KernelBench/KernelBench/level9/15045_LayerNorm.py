import torch
import torch.fft
import torch.nn
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, num_channels: 'int', eps: 'float'=1e-12):
        """Uses GroupNorm implementation with group=1 for speed."""
        super().__init__()
        self.layer_norm = torch.nn.GroupNorm(1, num_channels=num_channels,
            eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
