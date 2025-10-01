import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_ipt: 'int', n_head: 'int', dropout_p: 'float'=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.qkv_linear = nn.Linear(d_ipt, d_ipt * 3, True)
        self.n_head = n_head
        self.output_linear = nn.Linear(d_ipt, d_ipt, True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src: 'torch.FloatTensor', attn_mask: 'torch.FloatTensor'
        ) ->torch.FloatTensor:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        q, k, v = self.qkv_linear(src).chunk(3, dim=-1)
        q = q.contiguous().view(src.shape[0], src.shape[1], self.n_head, 
            src.shape[2] // self.n_head).permute(0, 2, 1, 3)
        k = k.contiguous().view(src.shape[0], src.shape[1], self.n_head, 
            src.shape[2] // self.n_head).permute(0, 2, 3, 1)
        v = v.contiguous().view(src.shape[0], src.shape[1], self.n_head, 
            src.shape[2] // self.n_head).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k)
        attn_weights = attn_weights * float(src.shape[2] // self.n_head
            ) ** -0.5
        attn_weights = attn_weights * attn_mask + (attn_mask - 1) * 10000.0
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(src
            .shape)
        attn_output = self.output_linear(attn_output)
        return attn_output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_ipt': 4, 'n_head': 4}]
