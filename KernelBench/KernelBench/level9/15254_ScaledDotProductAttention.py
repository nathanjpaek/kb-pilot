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


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
