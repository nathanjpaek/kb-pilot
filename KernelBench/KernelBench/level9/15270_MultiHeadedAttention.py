import torch
import numpy as np
import torch.utils.data


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled, softmax attention module for Transformer as defined by
    Attention(Q, K, V) on pg 4. Returns the final attention vectors as well as
    the attention matrices (pairwise scores). """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dropout=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / np.sqrt(K.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        scores = self.softmax(scores)
        if dropout is not None:
            scores = dropout(scores)
        return torch.matmul(scores, V), scores


class MultiHeadedAttention(torch.nn.Module):
    """
    Multi-headed attention layer for the Transformer model. Wraps
    ScaledDotProductAttention. Assumes n_heads are applied by splitting up
    model in to n_heads, each of size dm / n_heads. Guided by
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, dm, n_heads, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert dm % n_heads == 0, 'The dimension of the model must be evenly divisible by the number of attn heads.'
        self.dm = dm
        self.dk = dm // n_heads
        self.n_heads = n_heads
        self.wq = torch.nn.Linear(self.dm, self.dm)
        self.wk = torch.nn.Linear(self.dm, self.dm)
        self.wv = torch.nn.Linear(self.dm, self.dm)
        self.wo = torch.nn.Linear(self.dm, self.dm)
        self.attn_scores = None
        self.attn = ScaledDotProductAttention()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, preQ, preK, preV, mask=None):
        n_batch = preQ.shape[0]
        Q, K, V = self.wq(preQ), self.wk(preK), self.wv(preV)
        Q, K, V = (x.view(n_batch, -1, self.n_heads, self.dk).transpose(1, 
            2) for x in (Q, K, V))
        mask = mask.unsqueeze(1) if mask is not None else None
        attn_output, self.attn_scores = self.attn(Q, K, V, mask, self.dropout)
        attn_output = attn_output.transpose(1, 2).contiguous().view(n_batch,
            -1, self.dm)
        return self.wo(attn_output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dm': 4, 'n_heads': 4}]
