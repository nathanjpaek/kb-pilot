import torch
import torch.nn
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.nn as nn


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dropout': 0.5}]
