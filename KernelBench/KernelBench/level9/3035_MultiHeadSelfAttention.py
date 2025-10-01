import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed
import torch.distributions


def compute_attention(q, k, v, dropout=None, mask=None):
    """
    :param q: Query [B, NH, NQ, EL] or [NH, 1, EL] (in this case NQ=1)
    :param k: Key [B, NH, NK, EL]
    :param v: Value [B, NH, NK, EL]
    :param mask: [B, NQ, NK]
    :param dropout:
    :return:
    """
    if q.ndim + 1 == k.ndim:
        score = torch.einsum('nij,bnkj->bnik', q, k)
    elif q.ndim == k.ndim:
        score = torch.einsum('bnij,bnkj->bnik', q, k)
    score = score / np.sqrt(q.shape[-1])
    if mask is not None:
        mask = mask[:, None]
        score = score * mask + -100000000.0 * (1 - mask)
    score = F.softmax(score, dim=-1)
    if dropout is not None:
        score = dropout(score)
    return torch.einsum('bnij,bnjk->bnik', score, v)


class MultiHeadedAttentionBase(nn.Module):

    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        """
        :param embed_dim: The dimension of feature in each entity.
        :param num_heads: The number of attention heads.
        :param latent_dim:
        :param dropout:
        """
        super().__init__()
        self.w_k = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_v = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_o = nn.Parameter(torch.empty(num_heads, latent_dim, embed_dim))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)
        if hasattr(self, 'q'):
            nn.init.xavier_normal_(self.q)
        if hasattr(self, 'w_q'):
            nn.init.xavier_normal_(self.w_q)


class MultiHeadSelfAttention(MultiHeadedAttentionBase):

    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super().__init__(embed_dim, num_heads, latent_dim, dropout)
        self.w_q = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self._reset_parameters()

    def forward(self, x, mask=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, N, N] [batch size, length, length]
        :return: [B, N, E] [batch_size, length, embed_dim] Features after self attention
        """
        q = torch.einsum('blj,njd->bnld', x, self.w_q)
        k = torch.einsum('blj,njd->bnld', x, self.w_k)
        v = torch.einsum('blj,njd->bnld', x, self.w_v)
        out = compute_attention(q, k, v, self.dropout, mask)
        out = torch.einsum('bnlj,njk->blk', out, self.w_o)
        out = self.dropout(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'num_heads': 4, 'latent_dim': 4}]
