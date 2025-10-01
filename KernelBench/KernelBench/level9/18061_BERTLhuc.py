from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class BERTLhuc(nn.Module):

    def __init__(self, config):
        super(BERTLhuc, self).__init__()
        self.lhuc = Parameter(torch.zeros(config.hidden_size))

    def forward(self, hidden_states):
        hidden_states = hidden_states * 2.0 * nn.functional.sigmoid(self.lhuc)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
