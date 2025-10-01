import math
import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    score_i = f(Q, K_i), i = 1, 2, ..., t
        dot:        f(Q, K_i) = Q.transpose · K_i
        scaled_dot: f(Q, K_i) = Q.transpose · K_i / √(key_dim)
        general:    f(Q, K_i) = Q.transpose · W · K_i
        concat:     f(Q, K_i) = V.transpose · tanh(W · [Q; K_i])
        perceptron: f(Q, K_i) = V.transpose · tanh(W · Q + U · K_i)

    alpha_i = softmax(score_i)

    context = Σ(alpha_i · V_i)

    Args:
        query_dim: Dimension of query vector (Q).
        key_dim: Dimension of key vectors (K_i, i = 1, 2, ..., t).
        method: dot/scaled_dot/general/concat/perceptron
    """

    def __init__(self, query_dim, key_dim, value_dim=0, method='general',
        dropout_rate=0.0):
        super(Attention, self).__init__()
        self.method = method
        self.dropout = nn.Dropout(dropout_rate)
        if self.method == 'dot' or self.method == 'scaled_dot':
            assert query_dim == key_dim, 'The query_dim must equals key_dim.'
            if value_dim == 0:
                value_dim = key_dim
            self.linear_q = nn.Linear(query_dim, query_dim)
            self.linear_k = nn.Linear(key_dim, key_dim)
            self.linear_v = nn.Linear(value_dim, value_dim)
        elif self.method == 'general':
            self.W = nn.Linear(query_dim, key_dim, bias=False)
        elif self.method == 'concat':
            self.W = nn.Linear(query_dim + key_dim, query_dim + key_dim,
                bias=False)
            self.V = nn.Linear(query_dim + key_dim, 1, bias=False)
        elif self.method == 'perceptron':
            self.W = nn.Linear(query_dim, query_dim, bias=False)
            self.U = nn.Linear(key_dim, query_dim, bias=False)
            self.V = nn.Linear(query_dim, 1, bias=False)
        else:
            raise ValueError(
                'The method must be one of the following: dot, scaled_dot, general, concat or perceptron.'
                )

    def forward(self, queries, keys, values=None, mask=None, top_k=None):
        """
        Args:
            queries: Batch of query vectors (Q). Tensor[batch_size, query_len, query_dim]
            keys: Batch of key vectors (K_i, i = 1, 2, ..., t). Tensor[batch_size, key_len, key_dim]
            values: Batch of value vectors (V_i, i = 1, 2, ..., t). Tensor[batch_size, value_len, value_dim]
            mask: Use none zero value as valid flag and 0 as pad flag. Tensor[batch_size, query_len, key_len]
            top_k: Select top K relative values. int(0, ∞)

        Return:
            Batch of context vector (C). Tensor[batch_size, query_len, value_dim]
        """
        if values is None:
            values = keys
        else:
            assert values.shape[-2] == keys.shape[-2
                ], 'value_len Must equals key_len.'
        if self.method == 'dot' or self.method == 'scaled_dot':
            queries = self.linear_q(queries)
            keys = self.linear_k(keys)
            values = self.linear_v(values)
        scores = self.score(queries, keys)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
        alphas = F.softmax(scores, dim=-1)
        alphas = alphas.masked_fill(alphas != alphas, 0)
        if top_k is not None:
            _, indices = torch.topk(alphas, k=top_k, dim=-1, largest=True)
            self.get_device(alphas)
            topk_mask = torch.zeros(alphas.shape).scatter_(dim=-1, index=
                indices, src=torch.ones(indices.shape))
            alphas = alphas.masked_fill(topk_mask == 0, 0)
            alphas = F.softmax(alphas, dim=-1)
            alphas = alphas.masked_fill(alphas != alphas, 0)
        alphas = self.dropout(alphas)
        return torch.bmm(alphas, values)

    def score(self, queries, keys):
        """
        Args:
            queries: Tensor[batch_size, query_len, query_dim]
            keys: Tensor[batch_size, key_len, key_dim]

        Return:
            Batch of attention scores. Tensor[batch_size, query_len, key_len]
        """
        if self.method == 'dot' or self.method == 'scaled_dot':
            scores = torch.bmm(queries, keys.transpose(-1, -2))
            if self.method == 'scaled_dot':
                scores /= math.sqrt(keys.shape[-2])
            return scores
        elif self.method == 'general':
            return torch.bmm(self.W(queries), keys.transpose(-1, -2))
        elif self.method == 'concat':
            queries = queries.unsqueeze(2).expand(-1, -1, keys.shape[1], -1)
            keys = keys.unsqueeze(1).expand(-1, queries.shape[1], -1, -1)
            scores = torch.cat([queries, keys], dim=-1)
            scores = self.W(scores)
            scores = torch.tanh(scores)
            return self.V(scores).squeeze(3)
        elif self.method == 'perceptron':
            queries = queries.unsqueeze(2).expand(-1, -1, keys.shape[1], -1)
            keys = keys.unsqueeze(1).expand(-1, queries.shape[1], -1, -1)
            scores = self.W(queries) + self.U(keys)
            scores = torch.tanh(scores)
            return self.V(scores).squeeze(3)

    @staticmethod
    def get_device(t):
        try:
            device_id = t.get_device()
        except:
            return 'cpu'
        else:
            return 'cpu' if device_id < 0 else 'cuda'


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'query_dim': 4, 'key_dim': 4}]
