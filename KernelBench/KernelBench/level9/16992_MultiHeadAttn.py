import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttn(nn.Module):

    def __init__(self, d_model, n_head, dropout=0.1, scale=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        if scale:
            self.scale = math.sqrt(d_model // n_head)
        else:
            self.scale = 1

    def forward(self, x, mask):
        """

        :param x: bsz x max_len x d_model
        :param mask: bsz x max_len
        :return:
        """
        batch_size, max_len, _d_model = x.size()
        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)
        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        attn = torch.matmul(q, k)
        attn = attn / self.scale
        attn.masked_fill_(mask=mask[:, None, None].eq(0), value=float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)
        v = torch.matmul(attn, v)
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)
        v = self.fc(v)
        return v


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'n_head': 4}]
