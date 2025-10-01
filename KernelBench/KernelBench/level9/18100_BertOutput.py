from _paritybench_helpers import _mock_config
import torch
from torch import nn


class BertOutput(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.intermediate_size, model_config
            .hidden_size)
        self.LayerNorm = nn.LayerNorm(model_config.hidden_size, eps=
            model_config.layer_norm_eps)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model_config': _mock_config(intermediate_size=4,
        hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}]
