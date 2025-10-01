from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size ({}) is not a multiple of the number of attention heads ({})'
                .format(config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, query_hidden_states, context_hidden_states=None,
        attention_mask=None):
        mixed_query_layer = self.query(query_hidden_states)
        if context_hidden_states is None:
            mixed_key_layer = self.key(query_hidden_states)
            mixed_value_layer = self.value(query_hidden_states)
        else:
            mixed_key_layer = self.key(context_hidden_states)
            mixed_value_layer = self.value(context_hidden_states)
        batch_size, seq_length, _ = mixed_query_layer.size()
        query_layer = mixed_query_layer.view(batch_size, seq_length, self.
            num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        key_layer = mixed_key_layer.view(batch_size, seq_length, self.
            num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        value_layer = mixed_value_layer.view(batch_size, seq_length, self.
            num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(x + hidden_states)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self_attention = BertSelfAttention(config)
        self.self_output = BertSelfOutput(config)

    def forward(self, query_hidden_states, context_hidden_states=None,
        attention_mask=None):
        attention_output = self.self_attention(query_hidden_states,
            context_hidden_states, attention_mask)
        self_output = self.self_output(query_hidden_states, attention_output)
        return self_output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, num_attention_heads=
        4, attention_probs_dropout_prob=0.5, layer_norm_eps=1,
        hidden_dropout_prob=0.5)}]
