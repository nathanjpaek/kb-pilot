import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm(src)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
