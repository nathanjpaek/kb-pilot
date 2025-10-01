from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class BertMultiPairPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        hidden_states_first_cls = hidden_states[:, 0].unsqueeze(1).repeat([
            1, hidden_states.shape[1], 1])
        pooled_outputs = self.dense(torch.cat([hidden_states_first_cls,
            hidden_states], 2))
        pooled_outputs = self.activation(pooled_outputs)
        return pooled_outputs


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
