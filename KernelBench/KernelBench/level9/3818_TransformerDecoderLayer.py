import math
import torch
from torch import nn
import torch.nn.functional as F


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm
    """
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, f'Dimensions do not match: {dim} query vs {self.dim} configured'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            _bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads,
                dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(batch_size *
                n_heads, seq_len, dim_per_head)
            return tensor
        if key is None and value is None:
            key = value = query
        elif value is None:
            value = key
        _, key_len, dim = key.size()
        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))
        dot_prod = q.bmm(k.transpose(1, 2))
        attn_mask = (mask == 0).view(batch_size, 1, -1, key_len).repeat(1,
            n_heads, 1, 1).expand(batch_size, n_heads, query_len, key_len
            ).view(batch_size * n_heads, query_len, key_len)
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, -float(1e+20))
        attn_weights = F.softmax(dot_prod / scale, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attentioned = attn_weights.bmm(v)
        attentioned = attentioned.view(batch_size, n_heads, query_len,
            dim_per_head).transpose(1, 2).contiguous().view(batch_size,
            query_len, dim)
        out = self.out_lin(attentioned)
        return out


class TransformerFFN(nn.Module):

    def __init__(self, dim, dim_hidden, dropout=0):
        super(TransformerFFN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(self, n_heads, embedding_size, ffn_size, attention_dropout
        =0.0, relu_dropout=0.0, label_decoder=False):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.self_attention = MultiHeadAttention(n_heads, embedding_size,
            dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.encoder_attention = MultiHeadAttention(n_heads, embedding_size,
            dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, dropout=
            relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)
        self.label_decoder = label_decoder

    def forward(self, x, encoder_output, encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        residual = x
        x = self.self_attention(query=x, mask=decoder_mask)
        x = x + residual
        x = _normalize(x, self.norm1)
        residual = x
        x = self.encoder_attention(query=x, key=encoder_output, value=
            encoder_output, mask=encoder_mask)
        x = residual + x
        x = _normalize(x, self.norm2)
        residual = x
        x = self.ffn(x)
        x = residual + x
        x = _normalize(x, self.norm3)
        return x

    def _create_selfattn_mask(self, x):
        bsz = x.size(0)
        time = x.size(1)
        mask = torch.tril(x.new(time, time).fill_(1))
        if self.label_decoder:
            mask += x.new(time, time).fill_(1).triu(diagonal=2)
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_heads': 4, 'embedding_size': 4, 'ffn_size': 4}]
