import math
import torch
from torch import nn
from torch.functional import F
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, ..., q_len, q_dim): tensor containing projection vector for decoder.
        - **key** (batch, ..., k_len, k_dim): tensor containing features of the encoded input sequence. 
        - **value** (batch, ..., v_len, v_dim): tensor containing features of the encoded input sequence.
        - **mask** (batch, ..., q_len, k_len): tensor containing indices to be masked
        -  satisfy: q_dim = k_dim, v_len = k_len
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask=None):
        q_dim = query.size()[-1]
        k_dim = key.size()[-1]
        assert q_dim == k_dim
        score = torch.matmul(query, key.transpose(-2, -1))
        score = score / math.sqrt(k_dim)
        if mask is not None:
            score.masked_fill_(mask == 0, -float('Inf'))
        attn = F.softmax(score, -1)
        context = torch.matmul(attn, value)
        return context, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
