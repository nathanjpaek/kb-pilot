import torch
import torch.nn as nn


class MAXATTN(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
        add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MAXATTN, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, hidden, key=None, value=None):
        T = hidden.size(0)
        query = torch.max(hidden, dim=0, keepdim=True)[0]
        out, weight = self.attention_layer(query, hidden, hidden)
        return torch.cat([out for i in range(T)], dim=0), torch.cat([weight for
            i in range(T)], dim=1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'num_heads': 4}]
