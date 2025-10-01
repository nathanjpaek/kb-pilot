import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""

    def __init__(self, emb_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(emb_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(torch.tanh(self.proj(x)), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_size': 4, 'vocab_size': 4}]
