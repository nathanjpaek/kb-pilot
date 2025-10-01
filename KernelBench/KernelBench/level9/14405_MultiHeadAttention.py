import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from typing import Optional


def generate_local_map_mask(chunk_size: 'int', attention_size: 'int',
    mask_future=False, device: 'torch.device'='cpu') ->torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)
    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size
    return torch.BoolTensor(local_map)


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    """

    def __init__(self, d_model: 'int', q: 'int', v: 'int', h: 'int',
        attention_size: 'int'=None):
        """Initialize the Multi Head Block."""
        super().__init__()
        self._h = h
        self._attention_size = attention_size
        self._W_q = nn.Linear(d_model, q * self._h)
        self._W_k = nn.Linear(d_model, q * self._h)
        self._W_v = nn.Linear(d_model, v * self._h)
        self._W_o = nn.Linear(self._h * v, d_model)
        self._scores = None

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value:
        'torch.Tensor', mask: 'Optional[str]'=None) ->torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of ``'subsequent'``, None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        K = query.shape[1]
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(K)
        if self._attention_size is not None:
            attention_mask = generate_local_map_mask(K, self.
                _attention_size, mask_future=False, device=self._scores.device)
            self._scores = self._scores.masked_fill(attention_mask, float(
                '-inf'))
        if mask == 'subsequent':
            future_mask = torch.triu(torch.ones((K, K)), diagonal=1).bool()
            future_mask = future_mask
            self._scores = self._scores.masked_fill(future_mask, float('-inf'))
        self._scores = F.softmax(self._scores, dim=-1)
        attention = torch.bmm(self._scores, values)
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)
        self_attention = self._W_o(attention_heads)
        return self_attention

    @property
    def attention_map(self) ->torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                'Evaluate the model once to generate attention map')
        return self._scores


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_model': 4, 'q': 4, 'v': 4, 'h': 4}]
