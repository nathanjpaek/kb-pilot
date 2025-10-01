import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Norm(nn.Module):

    def __init__(self, dim_seq, input_size, eps=1e-06):
        super().__init__()
        self.size = input_size
        self.seq = dim_seq
        self.alpha = nn.Parameter(torch.ones((self.size, self.seq)))
        self.bias = nn.Parameter(torch.zeros((self.size, self.seq)))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim
            =-1, keepdim=True) + self.eps) + self.bias
        return norm


class Attention(nn.Module):

    def __init__(self, dim_seq, input_size, dropout=0.1):
        super().__init__()
        self.dim_seq = dim_seq
        self.dk = input_size
        self.q_linear = nn.Linear(dim_seq, dim_seq)
        self.k_linear = nn.Linear(dim_seq, dim_seq)
        self.v_linear = nn.Linear(dim_seq, dim_seq)
        self.norm_1 = Norm(dim_seq, input_size)
        self.norm_2 = Norm(dim_seq, input_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, s):
        s = self.norm_1(s).float()
        q = self.q_linear(s)
        k = self.k_linear(s)
        v = self.v_linear(s)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dk)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_1(scores)
        output = torch.matmul(scores, v)
        s = self.norm_2(s + self.dropout_2(output))
        return s


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_seq': 4, 'input_size': 4}]
