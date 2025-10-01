import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class FeedForward(nn.Module):

    def __init__(self, emb_dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(emb_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, emb_dim)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_dim': 4}]
