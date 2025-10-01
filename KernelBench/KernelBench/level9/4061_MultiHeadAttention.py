import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_state, num_heads=1):
        super().__init__()
        self.q_linear = nn.Linear(hidden_state, hidden_state)
        self.v_linear = nn.Linear(hidden_state, hidden_state)
        self.k_linear = nn.Linear(hidden_state, hidden_state)
        self.attention = nn.MultiheadAttention(hidden_state, num_heads)

    def forward(self, query_input, input, mask=None):
        query = self.q_linear(query_input)
        key = self.k_linear(input)
        value = self.v_linear(input)
        attn_output, _attn_output_weights = self.attention(query, key,
            value, mask)
        return attn_output


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_state': 4}]
