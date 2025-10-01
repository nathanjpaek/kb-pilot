import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention with optional positional encodings """

    def __init__(self, temperature, positional_encoding=None, attn_dropout=0.1
        ):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.positional_encoding = positional_encoding

    def forward(self, q, k, v, mask=None, dist_matrices=None):
        """
        q: [batch, heads, seq, d_k]  queries
        k: [batch, heads, seq, d_k]  keys
        v: [batch, heads, seq, d_v]  values
        mask: [batch, 1, seq, seq]   for each edge, which other edges should be accounted for. "None" means all of them.
                                mask is important when using local attention, or when the meshes are of different sizes.
        rpr: [batch, seq, seq, d_k]  positional representations
        """
        attn_k = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if self.positional_encoding is None:
            attn = attn_k
        else:
            attn_rpr = self.positional_encoding(q / self.temperature,
                dist_matrices)
            attn = attn_k + attn_rpr
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'temperature': 4}]
