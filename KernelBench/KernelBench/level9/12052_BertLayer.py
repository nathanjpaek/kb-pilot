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
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attn_value = torch.matmul(attention_probs, value)
        attn_value = attn_value.transpose(1, 2).contiguous()
        bs, seq_len = attn_value.shape[:2]
        attn_value = attn_value.view(bs, seq_len, -1)
        return attn_value

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
        proj = dense_layer(output)
        proj = dropout(proj)
        residual = proj + input
        ln = ln_layer(residual)
        return ln

    def forward(self, hidden_states, attention_mask):
        attn_outputs = self.self_attention(hidden_states, attention_mask)
        attn_outputs = self.add_norm(hidden_states, attn_outputs, self.
            attention_dense, self.attention_dropout, self.attention_layer_norm)
        proj = self.interm_dense(attn_outputs)
        proj = self.interm_af(proj)
        layer_output = self.add_norm(attn_outputs, proj, self.out_dense,
            self.out_dropout, self.out_layer_norm)
        return layer_output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(num_attention_heads=4, hidden_size=
        4, attention_probs_dropout_prob=0.5, layer_norm_eps=1,
        hidden_dropout_prob=0.5, intermediate_size=4)}]
