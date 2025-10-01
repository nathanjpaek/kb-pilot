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
        """
    x : bs * seq_len * hidden_size(word_embedding)
    fc : hidden_size * all_head_size
    x <- bs * seq_len * all_head_size(hidden_size)
    x <- bs * seq_len * (num_heads * head_size)
    x <- bs * num_heads * seq_len * head_size
    """
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.
            attention_head_size)
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        """ key, query, value: [bs, num_heads, seq_len, head_size] """
        score = query @ key.transpose(2, 3)
        score = score / math.sqrt(self.attention_head_size)
        """ score: [bs, num_heads, seq_len, seq_len] """
        score = score + attention_mask
        score = F.softmax(score, dim=3)
        score = self.dropout(score)
        score = score @ value
        """ score: [bs, num_heads, seq_len, head_size] """
        bs, num_attention_heads, seq_len, attention_head_size = score.shape
        score = score.view(bs, seq_len, num_attention_heads *
            attention_head_size)
        return score

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
