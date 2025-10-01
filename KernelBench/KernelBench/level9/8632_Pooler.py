from _paritybench_helpers import _mock_config
import torch
from torch import nn


class Pooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, last_hidden_state):
        pooled_output = self.dense(last_hidden_state)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
