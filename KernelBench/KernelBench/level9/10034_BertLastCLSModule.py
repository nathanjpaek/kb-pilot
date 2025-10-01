import torch
from torch import nn


class BertLastCLSModule(nn.Module):

    def __init__(self, dropout_prob=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input):
        last_hidden = input[-1][:, 0, :]
        out = self.dropout(last_hidden)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
