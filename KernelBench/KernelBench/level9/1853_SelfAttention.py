from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        attention_probs = torch.matmul(query_layer, key_layer.transpose(-1, -2)
            )
        attention_probs = nn.Softmax(dim=-1)(attention_probs)
        attention_scores = torch.sum(attention_probs, dim=-2)
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer, attention_scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
