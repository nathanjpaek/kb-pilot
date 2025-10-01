import torch
import torch.nn as nn
import torch._utils


class ScaleLayer(nn.Module):

    def __init__(self, init_value=1.0, lr_mult=1):
        super().__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(torch.full((1,), init_value / lr_mult,
            dtype=torch.float32))

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
