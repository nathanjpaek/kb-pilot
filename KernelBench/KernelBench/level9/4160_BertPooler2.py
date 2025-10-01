from _paritybench_helpers import _mock_config
import torch
from torch import nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import *
import torch.nn.functional


class BertPooler2(nn.Module):

    def __init__(self, config):
        super(BertPooler2, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4)}]
