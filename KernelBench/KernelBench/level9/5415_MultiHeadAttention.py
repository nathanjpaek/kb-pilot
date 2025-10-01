import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self._dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, pad_mask=None):
        """
        :param q: [bz, len_q, Q]
        :param k: [bz, len_k, K]
        :param v: [bz, len_v, V]
        :param pad_mask: [bz, len_q, len_k]  填充部分的mask
        more: Q==K, len_k==len_v
        :return: [bz, len_q, V]
        """
        att_weights = torch.matmul(q, k.transpose(-1, -2)).div(math.sqrt(k.
            size(-1)))
        if pad_mask is not None:
            att_weights.masked_fill_(pad_mask, -1000000000.0)
        soft_att_weights = self.softmax(att_weights)
        if self.training:
            soft_att_weights = self._dropout(soft_att_weights)
        att_out = torch.matmul(soft_att_weights, v)
        return att_out


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, nb_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._d_model = d_model
        self._d_k = d_k
        self._d_v = d_v
        self._nb_heads = nb_heads
        self._linear_qs = nn.Linear(in_features=d_model, out_features=d_k *
            nb_heads)
        self._linear_ks = nn.Linear(in_features=d_model, out_features=d_k *
            nb_heads)
        self._linear_vs = nn.Linear(in_features=d_model, out_features=d_v *
            nb_heads)
        self._linear_out = nn.Linear(in_features=d_v * nb_heads,
            out_features=d_model)
        self._self_attention = SelfAttention(dropout)
        self._dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self._linear_qs.weight, mean=0, std=math.sqrt(1 /
            self._d_model))
        nn.init.normal_(self._linear_ks.weight, mean=0, std=math.sqrt(1 /
            self._d_model))
        nn.init.normal_(self._linear_vs.weight, mean=0, std=math.sqrt(1 /
            self._d_model))
        nn.init.normal_(self._linear_out.weight, mean=0, std=math.sqrt(1 /
            self._d_model))

    def forward(self, q, k, v, att_mask=None):
        """
        :param q: [bz, len_q, d_model]
        :param k: [bz, len_k, d_model]
        :param v: [bz, len_v, d_model]
        :param att_mask: [bz, len_k]
        more: Q == K, len_k==len_v
        :return: [bz, len_q, d_model]
        """
        bz, len_q, _ = q.size()
        bz, len_k, _ = k.size()
        bz, len_v, _ = v.size()
        q_fc = self._linear_qs(q).reshape(bz, len_q, self._nb_heads, -1
            ).transpose(1, 2)
        k_fc = self._linear_ks(k).reshape(bz, len_k, self._nb_heads, -1
            ).transpose(1, 2)
        v_fc = self._linear_vs(v).reshape(bz, len_v, self._nb_heads, -1
            ).transpose(1, 2)
        if att_mask is not None:
            att_mask = att_mask[:, None, None, :]
        att_out = self._self_attention(q_fc, k_fc, v_fc, att_mask)
        att_out = att_out.transpose(1, 2).reshape(bz, len_q, -1)
        multi_head = self._linear_out(att_out)
        if self.training:
            multi_head = self._dropout(multi_head)
        return multi_head


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_k': 4, 'd_v': 4, 'nb_heads': 4}]
