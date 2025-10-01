import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import tensorboard as tensorboard


def attention(Q, K, V, mask, dropout=None):
    d_k = Q.size(-1)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)
    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))
    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)
    if dropout is not None:
        out = dropout(out)
    return out


class MultiheadedAttention(nn.Module):

    def __init__(self, d_model_Q, d_model_K, d_model_V, H, dout_p=0.0,
        d_model=None):
        super(MultiheadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p
        if self.d_model is None:
            None
            self.d_model = self.d_model_Q
        self.d_k = self.d_model // H
        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_model_Q)
        self.dropout = nn.Dropout(self.dout_p)
        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask):
        """ 
            Q, K, V: (B, Sq, Dq), (B, Sk, Dk), (B, Sv, Dv)
            mask: (B, 1, Sk)
            Sk = Sv, 
            Dk != self.d_k
            Also: m1 is the target modality (queries); m2 is the source modality (keys, values)
        """
        B, Sq, _d_model_Q = Q.shape
        Q = self.linear_Q2d(Q)
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)
        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        Q = attention(Q, K, V, mask, self.dropout)
        Q = Q.transpose(-3, -2).contiguous().view(B, Sq, self.d_model)
        Q = self.linear_d2Q(Q)
        return Q


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4,
        4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model_Q': 4, 'd_model_K': 4, 'd_model_V': 4, 'H': 4}]
