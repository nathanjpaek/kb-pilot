import torch
import numpy as np
from typing import Optional
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """Multihead scaled dot-product attention.
    """

    def __init__(self, contexts: 'int', queries: 'int', channels: 'int',
        heads: 'int'):
        """Initializer.
        Args:
            contexts: size of the key, value channels.
            queries: size of the query channels.
            channels: size of the hidden channels.
            heads: the number of the attnetion heads.
        """
        super().__init__()
        self.channels, self.heads = channels // heads, heads
        self.proj_key = nn.Linear(contexts, channels)
        self.proj_value = nn.Linear(contexts, channels)
        self.proj_query = nn.Linear(queries, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, query: 'torch.Tensor', key: 'torch.Tensor', value:
        'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Transform the inputs.
        Args:
            query: [torch.float32; [B, S, queries]], query.
            key: [torch.float32; [B, T, contexts]], key.
            value: [torch.float32; [B, T, contexts]], value.
            mask: [torch.float32; [B, S, T]], attention mask.
        Returns:
            [torch.float32; [B, S, C]], attended.
        """
        bsize, querylen, _ = query.shape
        keylen = key.shape[1]
        key = self.proj_key(key).view(bsize, keylen, self.heads, self.channels)
        value = self.proj_value(value).view(bsize, keylen, self.heads, self
            .channels)
        query = self.proj_query(query).view(bsize, querylen, self.heads,
            self.channels)
        score = torch.matmul(query.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1)
            ) * self.channels ** -0.5
        if mask is not None:
            score.masked_fill_(~mask[:, None, 0:1], -np.inf)
        weights = torch.softmax(score, dim=-1)
        out = torch.matmul(weights, value.transpose(1, 2))
        out = self.proj_out(out.transpose(1, 2).reshape(bsize, querylen, -1))
        if mask is not None:
            out = out * mask[..., 0:1]
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'contexts': 4, 'queries': 4, 'channels': 4, 'heads': 4}]
