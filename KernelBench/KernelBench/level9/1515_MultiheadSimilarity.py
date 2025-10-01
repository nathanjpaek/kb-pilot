import torch
import torch.utils.data
from torch import nn


class MultiheadSimilarity(nn.Module):

    def __init__(self, d_model, num_head, seq_len, in_proj=True):
        super().__init__()
        self.num_head = num_head
        self.seq_len = seq_len
        self.d_head = d_model // num_head
        self.in_proj = in_proj
        if self.in_proj:
            self.q_in_proj = nn.Linear(d_model, seq_len * d_model, bias=True)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(seq_len * d_model, d_model, bias=True)

    def forward(self, q, kv):
        bs, d_model = q.shape
        nbs = bs * self.num_head
        if self.in_proj:
            q_ = self.q_in_proj(q)
            q_ = q_.contiguous().view(bs, self.seq_len, d_model).transpose(0, 1
                )
            kv = q_ + kv
        q = self.q_proj(q)
        q = q.contiguous().view(nbs, self.d_head).unsqueeze(-1)
        k = self.k_proj(kv)
        k = k.contiguous().view(self.seq_len, nbs, self.d_head).transpose(0, 1)
        similarity = torch.bmm(k, q) * float(self.d_head) ** -0.5
        v = self.v_proj(kv)
        v = v.contiguous().view(self.seq_len, nbs, self.d_head).transpose(0, 1)
        v = (v * similarity).view(bs, self.num_head, self.seq_len, self.d_head)
        output = self.out_proj(v.flatten(1))
        return output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 1])]


def get_init_inputs():
    return [[], {'d_model': 4, 'num_head': 4, 'seq_len': 4}]
