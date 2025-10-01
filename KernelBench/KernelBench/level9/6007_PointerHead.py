import torch
import torch.quantization
from torch import nn


class PointerHead(nn.Module):
    """Head for pointer ordering task."""

    def __init__(self, embed_dim, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz, self.embed_dim
            ).transpose(0, 1)

    def forward(self, query, key):
        """Input shape: Time(SeqLen) x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        q = self._shape(q, tgt_len, bsz)
        k = self._shape(k, -1, bsz)
        assert k is not None
        assert q is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz, tgt_len, src_len)
        return attn_weights


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4}]
