import math
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn


class Attention(nn.Module):

    def __init__(self, dim, heads, max_len):
        super().__init__()
        self.q_mat = nn.Linear(dim, dim)
        self.k_mat = nn.Linear(dim, dim)
        self.v_mat = nn.Linear(dim, dim)
        self.dim = dim
        self.heads = heads
        self.max_len = max_len
        self.dk = dim // heads
        self.drop = nn.Dropout(0.1)
        self.softmax = nn.Softmax(-1)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        bs = x.size(0)
        q = self.q_mat(x).view(bs, -1, self.heads, self.dk)
        k = self.k_mat(x).view(bs, -1, self.heads, self.dk)
        v = self.v_mat(x).view(bs, -1, self.heads, self.dk)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dk)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(self.softmax(scores))
        output = torch.matmul(scores, v)
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.dim)
        output = self.out(concat)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'heads': 4, 'max_len': 4}]
