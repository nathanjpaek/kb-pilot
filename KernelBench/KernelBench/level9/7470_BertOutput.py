from _paritybench_helpers import _mock_config
import torch
import torch.nn
import torch.nn as nn


class BertOutput(nn.Module):
    """BERT output layer.
    Based on: BERT (pytorch-transformer)
    https://github.com/huggingface/transformers
    """

    def __init__(self, config) ->None:
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=
        0.5)}]
