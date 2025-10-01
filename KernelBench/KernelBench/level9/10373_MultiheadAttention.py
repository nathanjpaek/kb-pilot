from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn
from typing import Optional


class MultiheadAttention(nn.Module):
    """Multi-Head Attention Implemenetation from huggingface/transformer"""

    def __init__(self, config: 'ConveRTModelConfig'):
        super().__init__()
        self.num_attention_heads = 2
        self.num_attn_proj = config.num_embed_hidden
        self.attention_head_size = int(self.num_attn_proj / self.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.key = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.value = nn.Linear(config.num_embed_hidden, self.num_attn_proj)
        self.dropout = nn.Dropout(config.dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: 'torch.Tensor', attention_mask:
        'Optional[torch.Tensor]'=None, head_mask: 'Optional[torch.Tensor]'=
        None, encoder_hidden_states: 'Optional[torch.Tensor]'=None,
        encoder_attention_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            num_attn_proj,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(num_embed_hidden=4, dropout_rate=0.5)}]
