from _paritybench_helpers import _mock_config
import torch
from torch import nn


class BertPSIHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.decoder = nn.Linear(config.hidden_size, 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(2))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = hidden_states[:, 0]
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
