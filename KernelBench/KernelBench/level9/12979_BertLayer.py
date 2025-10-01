from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.
            attention_head_size)
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        att_scores = query @ key.transpose(-2, -1) / math.sqrt(self.
            attention_head_size)
        att_scores.masked_fill_(attention_mask == -10000.0, value=-10000.0)
        att_scores = F.softmax(att_scores, dim=-1)
        att_scores = self.dropout(att_scores)
        return att_scores @ value

    def forward(self, hidden_states, attention_mask):
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer,
            attention_mask)
        return attn_value


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size
            )
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=
            config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.interm_dense = nn.Linear(config.hidden_size, config.
            intermediate_size)
        self.interm_af = F.gelu
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size
            )
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
    input: the input
    output: the input that requires the sublayer to transform
    dense_layer, dropput: the sublayer
    ln_layer: layer norm that takes input+sublayer(output)
    """
        sublayer = dropout(dense_layer(output))
        return ln_layer(input + sublayer)

    def forward(self, hidden_states, attention_mask):
        attn_values = self.self_attention(hidden_states, attention_mask)
        bs = hidden_states.size(0)
        attn_values = attn_values.transpose(1, 2).contiguous().view(bs, -1,
            self.self_attention.all_head_size)
        hidden_states = self.add_norm(hidden_states, attn_values, self.
            attention_dense, self.attention_dropout, self.attention_layer_norm)
        interim_hidden_states = self.interm_af(self.interm_dense(hidden_states)
            )
        hidden_states = self.add_norm(hidden_states, interim_hidden_states,
            self.out_dense, self.out_dropout, self.out_layer_norm)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(num_attention_heads=4, hidden_size=
        4, attention_probs_dropout_prob=0.5, layer_norm_eps=1,
        hidden_dropout_prob=0.5, intermediate_size=4)}]
