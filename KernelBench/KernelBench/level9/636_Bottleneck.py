import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        residual = features
        features = self.layer_norm(features)
        features = self.w_2(F.relu(self.w_1(features)))
        features = self.dropout(features)
        features += residual
        return features


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_hid': 4}]
