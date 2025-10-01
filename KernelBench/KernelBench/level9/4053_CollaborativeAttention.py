import math
import torch
import torch.utils.data
from enum import Enum
import torch.nn as nn


class MixingMatrixInit(Enum):
    CONCATENATE = 1
    ALL_ONES = 2
    UNIFORM = 3


class CollaborativeAttention(nn.Module):

    def __init__(self, dim_input: 'int', dim_value_all: 'int',
        dim_key_query_all: 'int', num_attention_heads: 'int',
        mixing_initialization: 'MixingMatrixInit'=MixingMatrixInit.UNIFORM):
        super().__init__()
        if dim_value_all % num_attention_heads != 0:
            raise ValueError(
                'Value dimension ({}) should be divisible by number of heads ({})'
                .format(dim_value_all, num_attention_heads))
        self.dim_input = dim_input
        self.dim_value_all = dim_value_all
        self.dim_key_query_all = dim_key_query_all
        self.num_attention_heads = num_attention_heads
        self.mixing_initialization = mixing_initialization
        self.dim_value_per_head = dim_value_all // num_attention_heads
        self.attention_head_size = dim_key_query_all / num_attention_heads
        self.query = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.key = nn.Linear(dim_input, dim_key_query_all, bias=False)
        self.value = nn.Linear(dim_input, dim_value_all)
        self.mixing = self.init_mixing_matrix()

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None):
        from_sequence = hidden_states
        to_sequence = hidden_states
        if encoder_hidden_states is not None:
            to_sequence = encoder_hidden_states
            attention_mask = encoder_attention_mask
        query_layer = self.query(from_sequence)
        key_layer = self.key(to_sequence)
        mixed_query = query_layer[..., None, :, :] * self.mixing[..., :,
            None, :]
        mixed_key = key_layer[..., None, :, :]
        attention_scores = torch.matmul(mixed_query, mixed_key.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        value_layer = self.value(to_sequence)
        value_layer = self.transpose_for_scores(value_layer)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            dim_value_all,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def init_mixing_matrix(self, scale=0.2):
        mixing = torch.zeros(self.num_attention_heads, self.dim_key_query_all)
        if self.mixing_initialization is MixingMatrixInit.CONCATENATE:
            dim_head = int(math.ceil(self.dim_key_query_all / self.
                num_attention_heads))
            for i in range(self.num_attention_heads):
                mixing[i, i * dim_head:(i + 1) * dim_head] = 1.0
        elif self.mixing_initialization is MixingMatrixInit.ALL_ONES:
            mixing.one_()
        elif self.mixing_initialization is MixingMatrixInit.UNIFORM:
            mixing.normal_(std=scale)
        else:
            raise ValueError('Unknown mixing matrix initialization: {}'.
                format(self.mixing_initialization))
        return nn.Parameter(mixing)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_input': 4, 'dim_value_all': 4, 'dim_key_query_all': 4,
        'num_attention_heads': 4}]
