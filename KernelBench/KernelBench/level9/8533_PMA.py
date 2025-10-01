import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class MAB(nn.Module):

    def __init__(self, dim_X, dim_Y, dim, num_heads=4, ln=False, p=None):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_X, dim)
        self.fc_k = nn.Linear(dim_Y, dim)
        self.fc_v = nn.Linear(dim_Y, dim)
        self.fc_o = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.dropout1 = nn.Dropout(p=p) if p is not None else nn.Identity()
        self.dropout2 = nn.Dropout(p=p) if p is not None else nn.Identity()

    def forward(self, X, Y, mask=None):
        Q, K, V = self.fc_q(X), self.fc_k(Y), self.fc_v(Y)
        Q_ = torch.cat(Q.chunk(self.num_heads, -1), 0)
        K_ = torch.cat(K.chunk(self.num_heads, -1), 0)
        V_ = torch.cat(V.chunk(self.num_heads, -1), 0)
        A_logits = Q_ @ K_.transpose(-2, -1) / math.sqrt(Q.shape[-1])
        if mask is not None:
            mask = torch.stack([mask] * Q.shape[-2], -2)
            mask = torch.cat([mask] * self.num_heads, 0)
            A_logits.masked_fill_(mask, -float('inf'))
            A = torch.softmax(A_logits, -1)
            A.masked_fill_(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        attn = torch.cat((A @ V_).chunk(self.num_heads, 0), -1)
        O = self.ln1(Q + self.dropout1(attn))
        O = self.ln2(O + self.dropout2(F.relu(self.fc_o(O))))
        return O


class PMA(nn.Module):

    def __init__(self, dim_X, dim, num_inds, **kwargs):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(num_inds, dim))
        nn.init.xavier_uniform_(self.I)
        self.mab = MAB(dim, dim_X, dim, **kwargs)

    def forward(self, X, mask=None):
        I = self.I if X.dim() == 2 else self.I.repeat(X.shape[0], 1, 1)
        return self.mab(I, X, mask=mask)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim_X': 4, 'dim': 4, 'num_inds': 4}]
