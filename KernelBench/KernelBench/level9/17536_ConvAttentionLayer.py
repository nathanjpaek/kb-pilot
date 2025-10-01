import math
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.onnx.operators


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class ConvAttentionLayer(nn.Module):

    def __init__(self, c, hidden_size, dropout=0.0):
        super().__init__()
        self.in_projection = Linear(c, hidden_size)
        self.out_projection = Linear(hidden_size, c)
        self.dropout = dropout

    def forward(self, x, key, value, encoder_padding_mask=None,
        enc_dec_attn_constraint_mask=None):
        query = self.in_projection(x)
        attn_weights = torch.bmm(query.transpose(0, 1), key.transpose(0, 1)
            .transpose(1, 2))
        if encoder_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(encoder_padding_mask.
                unsqueeze(1), float('-inf')).type_as(attn_weights)
        if enc_dec_attn_constraint_mask is not None:
            attn_weights = attn_weights.masked_fill(
                enc_dec_attn_constraint_mask.bool(), float('-inf')).type_as(
                attn_weights)
        attn_logits = attn_weights
        sz = attn_weights.size()
        attn_scores = F.softmax(attn_weights.view(sz[0] * sz[1], sz[2]), dim=1)
        attn_scores = attn_scores.view(sz)
        attn_scores = F.dropout(attn_scores, p=self.dropout, training=self.
            training)
        attn = torch.bmm(attn_scores, value.transpose(0, 1)).transpose(0, 1)
        s = value.size(0)
        if encoder_padding_mask is None:
            attn = attn * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(attn).sum(dim=1, keepdim=True)
            s = s.transpose(0, 1).unsqueeze(-1)
            attn = attn * (s * s.rsqrt())
        attn = self.out_projection(attn)
        return attn, attn_scores, attn_logits


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'c': 4, 'hidden_size': 4}]
