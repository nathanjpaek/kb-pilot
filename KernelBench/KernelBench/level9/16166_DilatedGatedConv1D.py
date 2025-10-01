import torch
import torch.nn as nn


class DilatedGatedConv1D(nn.Module):

    def __init__(self, dilation_rate, dim):
        super().__init__()
        self.dim = dim
        self.dropout = nn.Dropout(p=0.1)
        self.cnn = nn.Conv1d(dim, dim * 2, 3, padding=dilation_rate,
            dilation=dilation_rate)

    def forward(self, x):
        residual = x
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x1, x2 = x[:, :, :self.dim], x[:, :, self.dim:]
        x1 = torch.sigmoid(self.dropout(x1))
        return residual * (1 - x1) + x2 * x1


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dilation_rate': 1, 'dim': 4}]
