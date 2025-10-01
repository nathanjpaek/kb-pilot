import math
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn


def gelu(x):
    """Implementation of the gelu activation function by Hugging Face"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class feedforward(nn.Module):

    def __init__(self, dim, heads, max_len):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        out = self.fc2(gelu(self.fc1(x)))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'heads': 4, 'max_len': 4}]
