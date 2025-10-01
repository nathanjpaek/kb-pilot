from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class ProteinResNetPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention_weights = nn.Linear(config.hidden_size, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None):
        attention_scores = self.attention_weights(hidden_states)
        if mask is not None:
            attention_scores += -10000.0 * (1 - mask)
        attention_weights = torch.softmax(attention_scores, -1)
        weighted_mean_embedding = torch.matmul(hidden_states.transpose(1, 2
            ), attention_weights).squeeze(2)
        pooled_output = self.dense(weighted_mean_embedding)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
