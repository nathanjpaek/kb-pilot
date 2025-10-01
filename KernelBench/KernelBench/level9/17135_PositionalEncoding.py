import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, dimension: 'int', dropout: 'float'=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dimension = dimension

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        position = torch.arange(x.shape[1]).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dimension, 2) * (-math.
            log(10000.0) / self.dimension))
        pe: 'torch.Tensor' = torch.zeros(1, x.shape[1], self.dimension).to(x
            .device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = torch.repeat_interleave(pe, x.shape[0], 0)
        x = torch.cat((x, pe[:x.shape[0]]), dim=-1)
        return self.dropout(x)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dimension': 4}]
