import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        assert query.size()[-1] == key.size()[-1]
        dim = query.size()[-1]
        tmp_raw_scores = torch.div(torch.matmul(query, key.transpose(-2, -1
            )), math.sqrt(dim))
        atte_weights = torch.softmax(tmp_raw_scores, dim=-1)
        atte_weights = self.dropout(atte_weights)
        output = torch.matmul(atte_weights, value)
        return output, atte_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, reduced_dim, n_head, dropout=0.0, eps
        =1e-08):
        super(MultiHeadAttention, self).__init__()
        assert reduced_dim % n_head == 0
        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.reduced_dim = reduced_dim
        self.Wq = nn.Linear(embedding_dim, reduced_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, reduced_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, reduced_dim, bias=False)
        self.inner_attention = ScaledDotProductAttention(dropout)
        self.Wo = nn.Linear(reduced_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embedding_dim, eps=eps)

    def forward(self, query):
        residual = query
        value = key = query
        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)
        b, n, _ = query.size()
        query = query.reshape(b, n, self.n_head, self.reduced_dim // self.
            n_head)
        b, m, _ = key.size()
        key = key.reshape(b, m, self.n_head, self.reduced_dim // self.n_head)
        value = value.reshape(b, m, self.n_head, self.reduced_dim // self.
            n_head)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, atte_weights = self.inner_attention(query, key, value)
        query = query.transpose(1, 2).reshape(b, n, self.reduced_dim)
        query = self.dropout(self.Wo(query))
        query = query + residual
        query = self.ln(query)
        return query, atte_weights


class ComprehensionLayer_step1(MultiHeadAttention):

    def __init__(self, embedding_dim, reduced_dim, n_head, dropout=0.0, eps
        =1e-08):
        super(ComprehensionLayer_step1, self).__init__(embedding_dim,
            reduced_dim, n_head, dropout)
        del self.ln
        self.low_ln = nn.LayerNorm(embedding_dim, eps=eps)
        self.mid_ln = nn.LayerNorm(embedding_dim, eps=eps)
        self.hig_ln = nn.LayerNorm(embedding_dim, eps=eps)

    def forward(self, low_vectors, mid_vectors, hig_vectors):
        b = low_vectors.size()[0]
        low_num, mid_num, hig_num = low_vectors.size()[1], mid_vectors.size()[1
            ], hig_vectors.size()[1]
        low_residual = low_vectors
        mid_residual = mid_vectors
        hig_residual = hig_vectors
        cated_vectors = torch.cat((low_vectors, mid_vectors, hig_vectors),
            dim=1)
        query = self.Wq(cated_vectors)
        key = self.Wk(cated_vectors)
        value = self.Wv(cated_vectors)
        low_query, mid_query, hig_query = torch.split(query, [low_num,
            mid_num, hig_num], dim=1)
        low_key, mid_key, hig_key = torch.split(key, [low_num, mid_num,
            hig_num], dim=1)
        low_value, mid_value, hig_value = torch.split(value, [low_num,
            mid_num, hig_num], dim=1)
        low_query = low_query.reshape(b, low_num, self.n_head, self.
            reduced_dim // self.n_head)
        low_key = low_key.reshape(b, low_num, self.n_head, self.reduced_dim //
            self.n_head)
        low_value = low_value.reshape(b, low_num, self.n_head, self.
            reduced_dim // self.n_head)
        low_query = low_query.transpose(1, 2)
        low_key = low_key.transpose(1, 2)
        low_value = low_value.transpose(1, 2)
        mid_query = mid_query.reshape(b, mid_num, self.n_head, self.
            reduced_dim // self.n_head)
        mid_key = mid_key.reshape(b, mid_num, self.n_head, self.reduced_dim //
            self.n_head)
        mid_value = mid_value.reshape(b, mid_num, self.n_head, self.
            reduced_dim // self.n_head)
        mid_query = mid_query.transpose(1, 2)
        mid_key = mid_key.transpose(1, 2)
        mid_value = mid_value.transpose(1, 2)
        hig_query = hig_query.reshape(b, hig_num, self.n_head, self.
            reduced_dim // self.n_head)
        hig_key = hig_key.reshape(b, hig_num, self.n_head, self.reduced_dim //
            self.n_head)
        hig_value = hig_value.reshape(b, hig_num, self.n_head, self.
            reduced_dim // self.n_head)
        hig_query = hig_query.transpose(1, 2)
        hig_key = hig_key.transpose(1, 2)
        hig_value = hig_value.transpose(1, 2)
        low_query, low_weights = self.inner_attention(low_query, low_key,
            low_value)
        mid_query, mid_weights = self.inner_attention(mid_query, mid_key,
            mid_value)
        hig_query, hig_weights = self.inner_attention(hig_query, hig_key,
            hig_value)
        low_query = low_query.transpose(1, 2).reshape(b, low_num, self.
            reduced_dim)
        mid_query = mid_query.transpose(1, 2).reshape(b, mid_num, self.
            reduced_dim)
        hig_query = hig_query.transpose(1, 2).reshape(b, hig_num, self.
            reduced_dim)
        output = self.dropout(self.Wo(torch.cat((low_query, mid_query,
            hig_query), dim=1)))
        low_vectors, mid_vectors, hig_vectors = torch.split(output, [
            low_num, mid_num, hig_num], dim=1)
        low_vectors = low_residual + low_vectors
        mid_vectors = mid_residual + mid_vectors
        hig_vectors = hig_residual + hig_vectors
        low_vectors = self.low_ln(low_vectors)
        mid_vectors = self.mid_ln(mid_vectors)
        hig_vectors = self.hig_ln(hig_vectors)
        return (low_vectors, mid_vectors, hig_vectors, low_weights,
            mid_weights, hig_weights)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'embedding_dim': 4, 'reduced_dim': 4, 'n_head': 4}]
