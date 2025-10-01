import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = torch.nn.Linear(in_features=d_model, out_features=d_key *
            n_head, bias=False)
        self.k_fc = torch.nn.Linear(in_features=d_model, out_features=d_key *
            n_head, bias=False)
        self.v_fc = torch.nn.Linear(in_features=d_model, out_features=
            d_value * n_head, bias=False)
        self.proj_fc = torch.nn.Linear(in_features=d_value * n_head,
            out_features=d_model, bias=False)

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:
            keys, values = queries, queries
            static_kv = False
        else:
            static_kv = True
        q = self.q_fc(queries)
        q = torch.reshape(q, shape=[q.size(0), q.size(1), self.n_head, self
            .d_key])
        q = q.permute(0, 2, 1, 3)
        if cache is not None and static_kv and 'static_k' in cache:
            k = cache['static_k']
            v = cache['static_v']
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            k = torch.reshape(k, shape=[k.size(0), k.size(1), self.n_head,
                self.d_key])
            k = k.permute(0, 2, 1, 3)
            v = torch.reshape(v, shape=[v.size(0), v.size(1), self.n_head,
                self.d_value])
            v = v.permute(0, 2, 1, 3)
        if cache is not None:
            if static_kv and 'static_k' not in cache:
                cache['static_k'], cache['static_v'] = k, v
            elif not static_kv:
                cache_k, cache_v = cache['k'], cache['v']
                k = torch.cat([cache_k, k], dim=2)
                v = torch.cat([cache_v, v], dim=2)
                cache['k'], cache['v'] = k, v
        return q, k, v

    def forward(self, queries, keys, values, attn_bias, cache=None):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q, k, v = self._prepare_qkv(queries, keys, values, cache)
        product = torch.matmul(q, k.transpose(2, 3))
        product = product * self.d_model ** -0.5
        if attn_bias is not None:
            product += attn_bias
        weights = F.softmax(product, dim=-1)
        if self.dropout_rate:
            weights = F.dropout(weights, p=self.dropout_rate)
        out = torch.matmul(weights, v)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, shape=[out.size(0), out.size(1), out.shape
            [2] * out.shape[3]])
        out = self.proj_fc(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 1, 4]), torch.rand([4, 4, 1, 4]), torch.rand(
        [4, 4, 1, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_key': 4, 'd_value': 4, 'd_model': 4}]
