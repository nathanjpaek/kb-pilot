import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def activation_factory(activation_type):
    if activation_type == 'RELU':
        return F.relu
    elif activation_type == 'TANH':
        return torch.tanh
    elif activation_type == 'ELU':
        return nn.ELU()
    else:
        raise ValueError('Unknown activation_type: {}'.format(activation_type))


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute a Scaled Dot Product Attention.

    Parameters
    ----------
    query
        size: batch, head, 1 (ego-entity), features
    key
        size: batch, head, entities, features
    value
        size: batch, head, entities, features
    mask
        size: batch,  head, 1 (absence feature), 1 (ego-entity)
    dropout

    Returns
    -------
    The attention softmax(QK^T/sqrt(dk))V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


class BaseModule(torch.nn.Module):
    """
    Base torch.nn.Module implementing basic features:
        - initialization factory
        - normalization parameters
    """

    def __init__(self, activation_type='RELU', reset_type='XAVIER'):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == 'XAVIER':
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == 'ZEROS':
                torch.nn.init.constant_(m.weight.data, 0.0)
            else:
                raise ValueError('Unknown reset type')
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

    def reset(self):
        self.apply(self._init_weights)


class EgoAttention(BaseModule):

    def __init__(self, feature_size=64, heads=4, dropout_factor=0):
        super().__init__()
        self.feature_size = feature_size
        self.heads = heads
        self.dropout_factor = dropout_factor
        self.features_per_head = int(self.feature_size / self.heads)
        self.value_all = nn.Linear(self.feature_size, self.feature_size,
            bias=False)
        self.key_all = nn.Linear(self.feature_size, self.feature_size, bias
            =False)
        self.query_ego = nn.Linear(self.feature_size, self.feature_size,
            bias=False)
        self.attention_combine = nn.Linear(self.feature_size, self.
            feature_size, bias=False)

    @classmethod
    def default_config(cls):
        return {}

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.feature_size),
            others), dim=1)
        key_all = self.key_all(input_all).view(batch_size, n_entities, self
            .heads, self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities,
            self.heads, self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.heads,
            self.features_per_head)
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1,
                self.heads, 1, 1))
        value, attention_matrix = attention(query_ego, key_all, value_all,
            mask, nn.Dropout(self.dropout_factor))
        result = (self.attention_combine(value.reshape((batch_size, self.
            feature_size))) + ego.squeeze(1)) / 2
        return result, attention_matrix


def get_inputs():
    return [torch.rand([4, 1, 64]), torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {}]
