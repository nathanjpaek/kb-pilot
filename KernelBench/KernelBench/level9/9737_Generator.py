import torch
import torch.nn as nn


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.proj1 = nn.Linear(self.d_model, self.d_model)
        self.proj = nn.Linear(self.d_model, vocab)

    def forward(self, x):
        sliced_x = x[:, 0, :]
        sliced_x = self.proj1(sliced_x)
        out = self.proj(sliced_x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'vocab': 4}]
