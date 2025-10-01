import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch._utils
import torch.nn


def activation(act_type='swish'):
    if act_type == 'swish':
        act = swish()
        return act
    else:
        act = nn.ReLU(inplace=True)
        return act


class swish(nn.Module):

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class GELU(nn.Module):

    @staticmethod
    def forward(x):
        erf = F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        return 0.5 * x * (1 + erf)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.1, activation=GELU):
        super(FeedForward, self).__init__()
        self.mlp1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.mlp2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'hidden_dim': 4}]
