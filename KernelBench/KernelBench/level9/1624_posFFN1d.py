import torch
from torch import nn


class posFFN1d(nn.Module):

    def __init__(self, d_hid, d_inner_hid, window=1, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, kernel_size=window)
        self.relu = nn.ReLU()
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, kernel_size=window)
        self.layer_norm = nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.w_1(x)
        out = self.relu(out)
        out = self.w_2(out)
        out = self.dropout(out)
        return self.layer_norm(out + x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_hid': 4, 'd_inner_hid': 4}]
