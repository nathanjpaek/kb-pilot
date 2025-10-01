import torch
import torch.nn as nn


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, size, vocab):
        super(Generator, self).__init__()
        self.size = size
        self.proj = nn.Linear(self.size, vocab)

    def forward(self, x):
        sliced_x = x[:, 0, :]
        out = self.proj(sliced_x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4, 'vocab': 4}]
