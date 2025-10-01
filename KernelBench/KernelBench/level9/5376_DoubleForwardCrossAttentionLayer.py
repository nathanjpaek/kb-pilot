import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResidualConnectionLayer(nn.Module):

    def __init__(self, dim_model, prob_dropout=0.1, add_sublayer=True):
        super(ResidualConnectionLayer, self).__init__()
        self.add_sublayer = add_sublayer
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x, sublayer):
        out = self.norm(x)
        out = sublayer(out)
        out = self.dropout(out)
        if self.add_sublayer:
            return x + out
        else:
            return out


class BaseLayer(nn.Module):

    def __init__(self, dim_model, dim_k, dim_v, h, dim_ff, prob_dropout):
        super(BaseLayer, self).__init__()
        self._dim_model = dim_model
        self._dim_k = dim_k
        self._dim_v = dim_v
        self._h = h
        self._dim_ff = dim_ff
        self._prob_dropout = prob_dropout


class MultiHeadedAttentionLayer(nn.Module):

    def __init__(self, dim_model, dim_k, dim_v, h):
        super(MultiHeadedAttentionLayer, self).__init__()
        self.dim_model = dim_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.h = h
        self.Q_linear = nn.Linear(dim_model, dim_k * h)
        self.K_linear = nn.Linear(dim_model, dim_k * h)
        self.V_linear = nn.Linear(dim_model, dim_v * h)
        self.out_linear = nn.Linear(self.h * dim_v, dim_model)

    def forward(self, Q, K, V, mask=None):
        b, len_q, len_k, len_v = Q.size(0), Q.size(1), K.size(1), V.size(1)
        Q_ = self.Q_linear(Q).view(b, len_q, self.h, self.dim_k).transpose(1, 2
            )
        K_ = self.K_linear(K).view(b, len_k, self.h, self.dim_k).transpose(1, 2
            )
        V_ = self.V_linear(V).view(b, len_v, self.h, self.dim_v).transpose(1, 2
            )
        if mask is not None:
            mask = mask.unsqueeze(1)
        out = self.__attention(Q_, K_, V_, mask)
        out = out.transpose(1, 2).contiguous().view(b, len_q, -1)
        out = self.out_linear(out)
        return out

    @staticmethod
    def __attention(Q, K, V, mask=None):
        d_k = K.shape[0]
        att = (Q / np.sqrt(d_k)).matmul(K.transpose(-1, -2))
        if mask is not None:
            att = att.masked_fill(mask == 0, -float('inf'))
        att = F.softmax(att, dim=-1)
        out = att.matmul(V)
        return out


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, dim_in, dim_ff, prob_dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim_in)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class DoubleForwardCrossAttentionLayer(BaseLayer):

    def __init__(self, dim_model, dim_k, dim_v, h, dim_ff, prob_dropout):
        super(DoubleForwardCrossAttentionLayer, self).__init__(dim_model,
            dim_k, dim_v, h, dim_ff, prob_dropout)
        self.self_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        self.rc1 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.context_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h
            )
        self.rc2 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.encoder_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h
            )
        self.rc3 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_model, dim_ff)
        self.rc4 = ResidualConnectionLayer(dim_model, prob_dropout)

    def forward(self, x, context_x, encoder_x, mask=None, context_mask=None,
        encoder_mask=None):
        out = self.rc1(x, lambda item: self.self_att(item, item, item, mask))
        out = self.rc2(out, lambda item: self.context_att(item, context_x,
            context_x, context_mask))
        out = self.rc3(out, lambda item: self.encoder_att(item, encoder_x,
            encoder_x, encoder_mask))
        out = self.rc4(out, self.ff)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'dim_model': 4, 'dim_k': 4, 'dim_v': 4, 'h': 4, 'dim_ff': 
        4, 'prob_dropout': 0.5}]
