import torch
from torch import nn


class PreprocessAtari(nn.Module):

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        return x / 255.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
