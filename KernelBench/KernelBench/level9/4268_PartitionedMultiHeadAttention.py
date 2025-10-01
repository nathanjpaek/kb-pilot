import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartitionedMultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, d_qkv, attention_dropout=0.1,
        initializer_range=0.02):
        super().__init__()
        self.w_qkv_c = nn.Parameter(torch.Tensor(n_head, d_model // 2, 3, 
            d_qkv // 2))
        self.w_qkv_p = nn.Parameter(torch.Tensor(n_head, d_model // 2, 3, 
            d_qkv // 2))
        self.w_o_c = nn.Parameter(torch.Tensor(n_head, d_qkv // 2, d_model //
            2))
        self.w_o_p = nn.Parameter(torch.Tensor(n_head, d_qkv // 2, d_model //
            2))
        bound = math.sqrt(3.0) * initializer_range
        for param in [self.w_qkv_c, self.w_qkv_p, self.w_o_c, self.w_o_p]:
            nn.init.uniform_(param, -bound, bound)
        self.scaling_factor = 1 / d_qkv ** 0.5
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)
        qkv_c = torch.einsum('btf,hfca->bhtca', x_c, self.w_qkv_c)
        qkv_p = torch.einsum('btf,hfca->bhtca', x_p, self.w_qkv_p)
        q_c, k_c, v_c = [c.squeeze(dim=3) for c in torch.chunk(qkv_c, 3, dim=3)
            ]
        q_p, k_p, v_p = [c.squeeze(dim=3) for c in torch.chunk(qkv_p, 3, dim=3)
            ]
        q = torch.cat([q_c, q_p], dim=-1) * self.scaling_factor
        k = torch.cat([k_c, k_p], dim=-1)
        v = torch.cat([v_c, v_p], dim=-1)
        dots = torch.einsum('bhqa,bhka->bhqk', q, k)
        if mask is not None:
            dots.data.masked_fill_(~mask[:, None, None, :], -float('inf'))
        probs = F.softmax(dots, dim=-1)
        probs = self.dropout(probs)
        o = torch.einsum('bhqk,bhka->bhqa', probs, v)
        o_c, o_p = torch.chunk(o, 2, dim=-1)
        out_c = torch.einsum('bhta,haf->btf', o_c, self.w_o_c)
        out_p = torch.einsum('bhta,haf->btf', o_p, self.w_o_p)
        return out_c, out_p


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'n_head': 4, 'd_qkv': 4}]
