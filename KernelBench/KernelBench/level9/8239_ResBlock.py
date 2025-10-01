import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, dim, dropout=0):
        super(ResBlock, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.dim, self.dim)
        self.linear2 = nn.Linear(self.dim, self.dim)
        self.layer_norm1 = nn.LayerNorm(self.dim)
        self.layer_norm2 = nn.LayerNorm(self.dim)
        self.reset_parameters()

    def reset_parameters(self):
        initScale = 0.1
        self.linear1.weight.data.uniform_(-initScale, initScale)
        self.linear1.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initScale, initScale)
        self.linear2.bias.data.zero_()

    def forward(self, x):
        x_prev = x
        x = self.layer_norm1(x)
        x = torch.tanh(x)
        x = self.linear1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x_prev + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
