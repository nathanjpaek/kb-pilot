from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import Softmax
from torch.nn import Linear


class Attention(nn.Module):

    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config['num_heads']
        self.attention_head_size = int(config['hidden_size'] / self.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = Linear(config['hidden_size'], self.all_head_size)
        self.key = Linear(config['hidden_size'], self.all_head_size)
        self.value = Linear(config['hidden_size'], self.all_head_size)
        self.out = Linear(self.all_head_size, config['hidden_size'])
        self.attn_dropout = Dropout(config['attention_dropout_rate'])
        self.proj_dropout = Dropout(config['attention_dropout_rate'])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(num_heads=4, hidden_size=4,
        attention_dropout_rate=0.5)}]
