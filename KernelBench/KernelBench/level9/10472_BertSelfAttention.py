from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn


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
        scores = torch.matmul(query, torch.transpose(key, 2, 3)) / math.sqrt(
            key.shape[-1])
        scores = scores.masked_fill(attention_mask < 0, -10000)
        normed = torch.softmax(scores, -1)
        per_head = torch.matmul(normed, value)
        atten = torch.cat([per_head[:, i, :, :] for i in range(per_head.
            shape[1])], -1)
        return atten

    def forward(self, hidden_states, attention_mask):
        """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer,
            attention_mask)
        return attn_value


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(num_attention_heads=4, hidden_size=
        4, attention_probs_dropout_prob=0.5)}]
