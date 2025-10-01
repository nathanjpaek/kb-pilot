import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProductDecoder(nn.Module):

    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.matmul(z, z.t()))
        return adj


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
